//===- LowerWorkdistribute.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering and optimisations of omp.workdistribute.
//
// Fortran array statements are lowered to fir as fir.do_loop unordered.
// lower-workdistribute pass works mainly on identifying fir.do_loop unordered
// that is nested in target{teams{workdistribute{fir.do_loop unordered}}} and
// lowers it to target{teams{parallel{distribute{wsloop{loop_nest}}}}}.
// It hoists all the other ops outside target region.
// Relaces heap allocation on target with omp.target_allocmem and
// deallocation with omp.target_freemem from host. Also replaces
// runtime function "Assign" with omp_target_memcpy.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include <clang/Parse/Parser.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsAttributes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <optional>
#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTETOJIT
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute-to-stablehlo"

using namespace mlir;

namespace {

using PrivateOmpOp = std::variant<omp::TeamsOp, 
                                  omp::ParallelOp, 
                                  omp::DistributeOp,
                                  omp::WsloopOp, 
                                  omp::SimdOp>;


static unsigned getBlockHostEvalVarsOffset(omp::TargetOp targetOp) {
  return targetOp.numHasDeviceAddrBlockArgs();
}

static unsigned getBlockMapVarsOffset(omp::TargetOp targetOp) {
  return targetOp.numHasDeviceAddrBlockArgs()
    + targetOp.numInReductionBlockArgs()
    + targetOp.numHostEvalBlockArgs();
}

static unsigned getReductionVarsOffset(Operation* wrapper) {
  auto offset = 0;
  llvm::TypeSwitch<Operation*>(wrapper)
    .Case([&](omp::WsloopOp wsLoopOp){
      // [private_vars, reduction_vars]
      offset = wsLoopOp.numPrivateBlockArgs();
      return;
    })
    .Case([&](omp::TeamsOp teamsOp){
      // [private_vars, reduction_vars]
      offset = teamsOp.numPrivateBlockArgs();
      return;
    })
    .Default([&](Operation*){return;});
  return offset;
}

/// Get wrappers ordering from outmost to innermost
static SmallVector<Operation*> getOmpWrappers(omp::TargetOp targetOp) {
  llvm::SmallVector<Operation*> wrappers;
  // The default order is from innermost to outermost
  targetOp.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::omp::TeamsOp,
                  mlir::omp::ParallelOp,
                  mlir::omp::DistributeOp,
                  mlir::omp::WsloopOp,
                  mlir::omp::SimdOp
      >(op)) {
      wrappers.push_back(op);
    }
  });
  // reverse so the wrappers is from outtermost to innermost
  std::reverse(wrappers.begin(), wrappers.end());
  return wrappers;
}

static std::optional<PrivateOmpOp> findInPrivateVars(Value val) {
  for (const auto& use: val.getUses()) {
    auto* owner = use.getOwner();
    if (auto teamsOp = llvm::dyn_cast<omp::TeamsOp>(owner)) {
      if (llvm::is_contained(teamsOp.getPrivateVars(), val)) {
        return teamsOp;
      }
    } else if (auto parallelOp = llvm::dyn_cast<omp::ParallelOp>(owner)) {
      if (llvm::is_contained(parallelOp.getPrivateVars(), val)) {
        return parallelOp; 
      }
    } else if (auto distributeOp = llvm::dyn_cast<omp::ParallelOp>(owner)) {
      if (llvm::is_contained(distributeOp.getPrivateVars(), val)) {
        return distributeOp; 
      }
    } else if (auto wsloopOp = llvm::dyn_cast<omp::WsloopOp>(owner)) {
      if (llvm::is_contained(wsloopOp.getPrivateVars(), val)) {
        return wsloopOp;
      }
    } else if (auto simdOp = llvm::dyn_cast<omp::SimdOp>(owner)) {
      if (llvm::is_contained(simdOp.getPrivateVars(), val)) {
        return simdOp;
      }
    }
  }
  return std::nullopt;
}

static void shadowGlobalVarWithLocalVar(PrivateOmpOp wrapper, Value globalVar, OpBuilder opBuilder) {
  std::visit([&](auto&& op){
    auto privateVars = op.getPrivateVars();
    auto it = llvm::find(privateVars, globalVar);
    if (it != privateVars.end()) {
    // showdow global var with local var
      unsigned idx = std::distance(privateVars.begin(), it);
      auto innerArg = op.getRegion().front().getArgument(idx);

      hlfir::DeclareOp innerDOp = nullptr;
      Value localVal = nullptr;
      for (Operation* user: innerArg.getUsers()) {
        if (auto innerDeclareOp = llvm::dyn_cast<hlfir::DeclareOp>(user)) {
          innerDOp = innerDeclareOp;
          localVal = innerDeclareOp.getBase();
        }
      }
      if (!innerDOp || !localVal) {
        return;
      }
      globalVar.replaceUsesWithIf(localVal, [&](OpOperand& globalUse)-> bool{
        return op->isAncestor(globalUse.getOwner());
      });
      
      // showdow local var with alloca var
      auto savedPtr = opBuilder.saveInsertionPoint(); 
      opBuilder.setInsertionPoint(innerDOp);
      auto argType = innerArg.getType();
      auto eleType = argType;
      if (auto refType = llvm::dyn_cast<fir::ReferenceType>(argType)) {
        eleType =  refType.getEleTy();
      }
      auto allocaOp = fir::AllocaOp::create(opBuilder, innerDOp->getLoc(), eleType);
      innerDOp.getMemrefMutable().assign(allocaOp.getResult());

      // erase the mapping
      op.getPrivateVarsMutable().erase(idx);
      if (auto syms = op.getPrivateSyms()) {
        llvm::SmallVector<mlir::Attribute> newSyms(syms->getValue());
        newSyms.erase(newSyms.begin() + idx);
        if (newSyms.empty()) {
          op.removePrivateSymsAttr();
      } else {
          op.setPrivateSymsAttr(opBuilder.getArrayAttr(newSyms));
        }
      }
      op.getRegion().front().eraseArgument(idx);
      opBuilder.restoreInsertionPoint(savedPtr);
    }
  }, wrapper);
}

/// Before:
///   %32 = declare(%arg1)
///   omp.simd private (%32#0 -> %arg11) {
///     %80 = declare (%arg11)
///     ...use of %80
///     store value to %32#0
///   }
///
/// After Step 1 (replace all usage of global private variables):
///   %32 = declare(%arg1)
///   omp.simd private (%32#0 -> %arg11) {
///     %80 = declare (%arg11)
///     ... use of %80
///     store value to %80#0 // change here
///   }
///
/// After Step 2 (allocate another local value to shadow original local variable):
///   %32 = declare(%arg1)
///   omp.simd private (%32#0 -> %arg11) {
///     %alloca = alloca (mem) // add here
///     %alloca80 = declare (alloca) // add here
///     %80 = declare (%arg11)
///     ... use of %alloca80 // change here
///     store value to %alloca#0 // change here
///   }
/// 
/// After Step 3 (delete the private and global):
///   omp.simd {
///     %alloca = alloca (mem) 
///     %alloca80 = declare (alloca)
///     ... use of %alloca80 
///     store value to %alloca#0
///   }
static void removePrivateVarsFromMapEntry(
  omp::TargetOp targetOp,
  OpBuilder& opBuilder,
  llvm::SmallVector<int>& promotedArgs
) {
  auto& entryBlock = targetOp.getRegion().front();
  auto mapVarsOffSet = getBlockMapVarsOffset(targetOp);
  
  auto updatePromotedArgs = [&](int rmIdx) -> void {
    for (size_t i = 0; i < promotedArgs.size(); i++) {
      if (promotedArgs[i] == rmIdx) {
        promotedArgs.erase(promotedArgs.begin() + i);
      } else if (promotedArgs[i] > rmIdx) {
        promotedArgs[i] -= 1;
      } else {
        // promotedArgs[i] < rmIdx, do nothing
      }
    }
  };

  for (int i = targetOp.numMapBlockArgs() - 1; i >= 0; i--) {
    auto argIdx = i + mapVarsOffSet;
    auto arg = entryBlock.getArgument(argIdx);
    
    if (llvm::range_size(arg.getUsers()) != 1) {
      continue;
    }

    hlfir::DeclareOp declareOp = nullptr;
    for (auto* userOp: arg.getUsers()) {
      if (!llvm::isa<hlfir::DeclareOp>(userOp)) {
        continue;
      }
      declareOp = llvm::cast<hlfir::DeclareOp>(userOp); 
      assert(declareOp);
    }
    if (!declareOp) continue;

    auto targetDeclaredOperand = declareOp.getBase();
    std::optional<PrivateOmpOp> found = findInPrivateVars(targetDeclaredOperand);
    if (!found.has_value()) continue;

    shadowGlobalVarWithLocalVar(found.value(), targetDeclaredOperand, opBuilder);
    assert(declareOp.getResults().use_empty());
    declareOp.erase();
    arg.dropAllUses();
    entryBlock.eraseArgument(argIdx);
    targetOp.getMapVarsMutable().erase(i);

    updatePromotedArgs(argIdx);
  }
}

static void expandLoopNestOp(
  omp::LoopNestOp lNOp,
  OpBuilder& opBuilder
) {
  auto saved = opBuilder.saveInsertionPoint();

  fir::DoLoopOp outmostLoopOp = nullptr;
  fir::DoLoopOp lastLoopOp = nullptr;
  llvm::SmallVector<mlir::Value> newInductionVars;
  // from outmost to innermost
  for (uint64_t i = 0; i < lNOp.getCollapseNumLoops(); i ++) {
    if (!outmostLoopOp) {
      opBuilder.setInsertionPoint(lNOp);
    } else {
      opBuilder.setInsertionPointToStart(lastLoopOp.getBody());
    }
    auto convertToIndexIfNot = [&](Value val) -> Value {
      if (!val.getType().isIndex()) {
        auto convertOp = fir::ConvertOp::create(opBuilder, lNOp.getLoc(), IndexType::get(opBuilder.getContext()), val, {});
        return convertOp.getResult();
      }
      return val;
    };
    auto lb = convertToIndexIfNot(lNOp.getLoopLowerBounds()[i]);
    auto ub = convertToIndexIfNot(lNOp.getLoopUpperBounds()[i]);
    auto step = convertToIndexIfNot(lNOp.getLoopSteps()[i]);

    auto loopOp = fir::DoLoopOp::create(opBuilder, lNOp.getLoc(), lb, ub, step);
    if (outmostLoopOp == nullptr) {
      outmostLoopOp = loopOp;
    }
    lastLoopOp = loopOp;
    newInductionVars.push_back(loopOp.getInductionVar()); 
  }

  assert(lastLoopOp);

  auto* targetBlock = lastLoopOp.getBody();
  auto* sourceBlock = &lNOp.getRegion().front();

  for (uint64_t i = 0; i < lNOp.getCollapseNumLoops(); i++) {
    auto loopIV = newInductionVars[i];
    auto lNOpIV = lNOp.getRegion().front().getArgument(i);
    if (loopIV.getType() != lNOpIV.getType()) {
      opBuilder.setInsertionPointToStart(lastLoopOp.getBody());
      auto loopIVConvertOp = fir::ConvertOp::create(opBuilder, lNOp.getLoc(), lNOpIV.getType(), loopIV, {});
      loopIV = loopIVConvertOp.getResult();
    }
    sourceBlock->getArgument(i).replaceAllUsesWith(loopIV);
  }

  mlir::Block::iterator insertPt(targetBlock->getTerminator());
  targetBlock->getOperations().splice(
    insertPt,
    sourceBlock->getOperations(),
    sourceBlock->getOperations().begin(),
    std::prev(sourceBlock->getOperations().end()) 
  );
  lNOp.erase();

  opBuilder.restoreInsertionPoint(saved);
  assert(outmostLoopOp);
  return;
}


template<typename T>
static void replaceLocalReductionVars(T wrapper) {
  auto reductionVarsOffset = getReductionVarsOffset(wrapper);
  auto& entryBlock = wrapper.getRegion().front();

  int reductionVarsCount = wrapper.numReductionBlockArgs();
  assert(wrapper.getReductionVars().size() == reductionVarsCount);

  for (int i = 0; i < reductionVarsCount; i++) {
    Value reductionVarArg = entryBlock.getArgument(reductionVarsOffset + i);
    Value reductionVar = wrapper.getReductionVars()[i];

    for (auto &use: reductionVarArg.getUses()) {
      if (auto declaredOp = llvm::dyn_cast<hlfir::DeclareOp>(use.getOwner())) {
        assert(declaredOp.getMemref() == reductionVarArg);
        declaredOp.getResult(0).replaceAllUsesWith(reductionVar);
        if (declaredOp.getNumResults() > 1) {
          assert(declaredOp.getNumResults() == 2);
          assert(declaredOp.getResult(1).use_empty());
        }
        assert(declaredOp.use_empty());
        declaredOp.erase();
      } else {
        use.assign(reductionVar);
      }
    }
    assert(reductionVarArg.use_empty());
  }

  for (int i = reductionVarsCount - 1; i >= 0; i--) {
    entryBlock.eraseArgument(reductionVarsOffset + i);
    wrapper.getReductionVarsMutable().erase(i);
  }

  assert(wrapper.getReductionVars().empty());

  if (wrapper->hasAttr(wrapper.getReductionSymsAttrName())) {
    wrapper->removeAttr(wrapper.getReductionSymsAttrName());
  }
}

template<typename T, typename Callable>
static bool applyToConcreteType(Operation* wrapper, Callable f) {
  if (auto concreteOp = llvm::dyn_cast<T>(wrapper)) {
    f(concreteOp);
    return true;
  }
  return false;
}

static void flattenTargetOp(
  llvm::SmallVector<Operation*>& wrappers, 
  omp::TargetOp targetOp, 
  OpBuilder& builder
) {
  // erase reduction vars
  for (int i = wrappers.size() - 1; i >= 0; i--) {
    auto* wrapper = wrappers[i]; 
    if (
      applyToConcreteType<omp::WsloopOp>(wrapper, replaceLocalReductionVars<omp::WsloopOp>)
      || applyToConcreteType<omp::TeamsOp>(wrapper, replaceLocalReductionVars<omp::TeamsOp>)
    ) {
      continue;
    }
  }

  // expand loopNests
  llvm::SmallVector<omp::LoopNestOp> loopNests;
  targetOp.walk([&](omp::LoopNestOp lNOp) {
    loopNests.push_back(lNOp);
  });
  for (auto& lNOp: loopNests) {
    expandLoopNestOp(lNOp, builder);
  }

  for (size_t i = 0; i < wrappers.size(); i++) {
    auto* wrapper = wrappers[i]; 
    if (wrapper->getNumRegions() == 0 || wrapper->getRegion(0).empty()) {
      continue;
    }
    mlir::Block *parentBlock = wrapper->getBlock();
    mlir::Block *innerBlock = &wrapper->getRegion(0).front();
    auto &innerOps = innerBlock->getOperations();
    mlir::Block::iterator insertPt(wrapper);
    auto insertEnd = innerOps.end();
    if (!innerOps.empty() && innerOps.back().hasTrait<OpTrait::IsTerminator>()) {
      insertEnd = mlir::Block::iterator(&innerOps.back());
    }
    parentBlock->getOperations().splice(
      insertPt,
      innerOps,
      innerOps.begin(),
      insertEnd
    );
    wrapper->erase();
  }
}


// `host_eval_vars` contains info like loop bounds, they will be propagated during the device compilation.
// The problem is that it will be promoted as argument, but not really mapped, leading the problem that the JITCode function have more arguments than NumArgs we get. 
static void absorbHostEvalVarsToMapEntries(
  omp::TargetOp targetOp, 
  OpBuilder& opBuilder,
  llvm::SmallVector<int>& indicesForAbsorbedHostEvalVars
) {
  auto& entryBlock = targetOp.getRegion().front();

  auto hostEvalOffset = getBlockHostEvalVarsOffset(targetOp); 
  unsigned originalHostEvalVarsCount = targetOp.numHostEvalBlockArgs();

  llvm::SmallVector<unsigned> indicesForAbsorbedToBody;
  llvm::SmallVector<unsigned> indicesForAbsorbedToMapVars;
  for (unsigned i = 0; i < targetOp.numHostEvalBlockArgs(); i++) {
    // TODO: we can fold some constant arguments so not necessarily migrate all hostEvalIndices.
    auto arg = targetOp.getHostEvalVars()[i];
    if (matchPattern(arg, m_Constant())) {
      indicesForAbsorbedToBody.push_back(i);
    } else {
      indicesForAbsorbedToMapVars.push_back(i);
    }
  }

  OpBuilder::InsertionGuard guard(opBuilder);

  // absorb some constant values directly into the target body
  opBuilder.setInsertionPointToStart(&entryBlock);
  for (const unsigned& idx: indicesForAbsorbedToBody) {
    auto hostEvalVar = targetOp.getHostEvalVars()[idx];
    auto hostEvalVarArg = entryBlock.getArgument(hostEvalOffset + idx);
    assert(hostEvalVar.getType().isInteger());
    IntegerAttr attr;
    if (matchPattern(hostEvalVar, m_Constant(&attr))) {
      auto constOp = arith::ConstantOp::create(opBuilder, targetOp.getLoc(), hostEvalVar.getType(), attr); 
      hostEvalVarArg.replaceAllUsesWith(constOp.getResult());
    } else {
      llvm::errs() << "Should be pattern matched!\n";
      std::exit(EXIT_FAILURE);
    }
  }

  // Append HostEvalVars to MapVars and replace uses.
  auto mapVarsOffset = getBlockMapVarsOffset(targetOp);
  assert(mapVarsOffset >= hostEvalOffset + targetOp.numHostEvalBlockArgs());
  for (const unsigned& idx : indicesForAbsorbedToMapVars) {
    auto hostEvalVar = targetOp.getHostEvalVars()[idx];
    // MapVars only accept ref args
    // hostEvalVar usually type i32, but we need to assert
    assert(hostEvalVar.getType().isInteger());
    opBuilder.setInsertionPoint(targetOp);
    auto preAllocaOp = fir::AllocaOp::create(opBuilder, targetOp.getLoc(), hostEvalVar.getType());
    fir::StoreOp::create(opBuilder, targetOp.getLoc(), hostEvalVar, preAllocaOp.getResult());
    auto hostEvalVarRef = omp::MapInfoOp::create(
      opBuilder, 
      targetOp.getLoc(), 
      preAllocaOp.getType(), 
      preAllocaOp, 
      TypeAttr::get(hostEvalVar.getType()),
      /*ClauseMapFlagsAttr*/omp::ClauseMapFlagsAttr::get(targetOp.getContext(), ::mlir::omp::ClauseMapFlags::implicit),
      /*VariableCaptureKindAttr*/omp::VariableCaptureKindAttr::get(targetOp.getContext(), ::mlir::omp::VariableCaptureKind::ByCopy),
      /*opational: var_ptr_ptr*/{},
      /*ValueRange members*/{},
      /*ArrayAttr members_index*/{},
      /*ValueRange bounds*/{},
      /*FlatSymbolRefAttr mapper_id*/{},
      /*name*/StringAttr::get(targetOp.getContext()),
      /*partial_map=*/BoolAttr::get(targetOp.getContext(), false)
    );

    int insertedArgPosition = mapVarsOffset + targetOp.numMapBlockArgs();
    auto insertedArg = entryBlock.insertArgument(insertedArgPosition, hostEvalVarRef.getResult().getType(), targetOp.getLoc());
    targetOp.getMapVarsMutable().append(hostEvalVarRef.getResult());
    // recover from fir.ref<i32> to i32 and replace original usage of hostEvalVars
    opBuilder.setInsertionPointToStart(&entryBlock);
    auto insertedArgVal = fir::LoadOp::create(opBuilder, targetOp.getLoc(), insertedArg);
    entryBlock.getArgument(idx + hostEvalOffset).replaceAllUsesWith(insertedArgVal);

    indicesForAbsorbedHostEvalVars.push_back(insertedArgPosition);
  }

  for (int i = originalHostEvalVarsCount - 1; i >= 0; i--) {
    targetOp.getHostEvalVarsMutable().erase(i);
    entryBlock.eraseArgument(hostEvalOffset + i);
  }
  for (size_t i = 0; i < indicesForAbsorbedHostEvalVars.size(); i++) {
    indicesForAbsorbedHostEvalVars[i] -= originalHostEvalVarsCount;
  }
}

class LowerWorkdistributeToJitPass
    : public flangomp::impl::LowerWorkdistributeToJitBase<
          LowerWorkdistributeToJitPass> {
public:
  void runOnOperation() override {
    MLIRContext &context = getContext();
    auto moduleOp = getOperation();
    OpBuilder opBuilder(&context);

    SmallVector<omp::TargetOp> targetOps;
    moduleOp.walk(
        [&](omp::TargetOp targetOp) { targetOps.push_back(targetOp); });
    for (auto targetOp : targetOps) {
      // Block argument indices of arguments been promoted from HostEvalVars to MapVars
      llvm::SmallVector<int> indicesForAbsorbedHostEvalVars;

      if (targetOp->walk([](omp::WorkdistributeOp) { return WalkResult::interrupt();}).wasInterrupted()) {
        auto &region = targetOp.getRegion();
        auto &block = region.front();
        auto mapVarsMutable = targetOp.getMapVarsMutable();
        int mapEntrySize = mapVarsMutable.size();
        for (int i = mapEntrySize - 1; i >= 0; i--) {
          auto arg = block.getArgument(i);
          // `omp.target` will try to map the outside declaration of all variables used in the omp region,
          // even if it is declared as private here. 
          // In such case, it will be shadowed by the `fir::alloca` declared inside of targetOp region,
          // becoming dead code. We need to clean it up so the host will not try to alloc memory and transferring to device.
          bool isShadowedByPrivateVar = true; 
          for (auto* userOp : arg.getUsers()) {
            auto dOp = llvm::dyn_cast<hlfir::DeclareOp>(userOp);
            if (dOp) {
              // If one of its result get used, can not delete
              if (!dOp.getResult(0).use_empty() || !dOp.getResult(1).use_empty()) {
                isShadowedByPrivateVar = false;
                break;
              }
            } else {
              isShadowedByPrivateVar = false;
              break;
            }
          }

          if (isShadowedByPrivateVar) {
            // Delete the extra declare before delete the arg itself
            for (auto* userOp: arg.getUsers()) {
              userOp->erase();
            } 

            block.eraseArgument(i);
            mapVarsMutable.erase(i);
          }
        }
      } else if (targetOp.walk([&](omp::TargetOp top){if (top != targetOp) {return WalkResult::interrupt();} return WalkResult::advance();}).wasInterrupted()==false) {
        absorbHostEvalVarsToMapEntries(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        removePrivateVarsFromMapEntry(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        auto wrappers = getOmpWrappers(targetOp);
        flattenTargetOp(wrappers, targetOp, opBuilder);
      } else {
        LDBG() << "Ignoring non-workdistribute and nested target op:\n" << *targetOp;
        continue;
      }

      std::string str;
      {
        OpBuilder b(&context);
        b.setInsertionPointToStart(moduleOp.getBody());
        auto f = func::FuncOp::create(
            b, targetOp.getLoc(), "kernel",
            FunctionType::get(&context, targetOp.getRegion().begin()->getArgumentTypes(), {}));
        b.cloneRegionBefore(targetOp.getRegion(), f.getRegion(), f.getRegion().begin());
        for (const auto& argIdx: indicesForAbsorbedHostEvalVars) {
          f.setArgAttr(argIdx, "llvm.noalias", b.getUnitAttr());
          f.setArgAttr(argIdx, "llvm.readonly", b.getUnitAttr());
        }

        llvm::raw_string_ostream os(str);
        os << *f;
        f->erase();
      }

      OpBuilder b(targetOp);
      auto targetJitOp = omp::TargetJitOp::create(
          b, targetOp.getLoc(), targetOp.getAllocateVars(),
          targetOp.getAllocatorVars(), targetOp.getBareAttr(),
          targetOp.getDependKindsAttr(), targetOp.getDependVars(),
          targetOp.getDependIteratedKindsAttr(), targetOp.getDependIterated(),
          targetOp.getDevice(), targetOp.getHasDeviceAddrVars(),
          targetOp.getHostEvalVars(), targetOp.getIfExpr(),
          targetOp.getInReductionVars(), targetOp.getInReductionByrefAttr(),
          targetOp.getInReductionSymsAttr(), targetOp.getIsDevicePtrVars(),
          targetOp.getMapVars(), targetOp.getNowaitAttr(),
          targetOp.getPrivateVars(), targetOp.getPrivateSymsAttr(),
          targetOp.getPrivateNeedsBarrierAttr(), targetOp.getThreadLimitVars(),
          targetOp.getPrivateMapsAttr(),
          /*jit_code=*/StringAttr::get(&context, str.c_str()));
      targetJitOp.getRegion().takeBody(targetOp.getRegion());
      targetJitOp.getRegion().begin()->clear();
      b.setInsertionPointToStart(&*targetJitOp.getRegion().begin());
      omp::TerminatorOp::create(b, targetOp->getLoc());
      targetOp->erase();
    }
  }
};
} // namespace
