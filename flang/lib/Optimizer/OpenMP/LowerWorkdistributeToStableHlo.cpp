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
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTETOJIT
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute-to-stablehlo"

using namespace mlir;

namespace {

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

static bool inPrivateVars(Value val) {
  for (const auto& use: val.getUses()) {
    auto* owner = use.getOwner();
    if (auto teamsOp = llvm::dyn_cast<omp::TeamsOp>(owner)) {
      if (llvm::is_contained(teamsOp.getPrivateVars(), val)) {
        return true;
      }
    } else if (auto parallelOp = llvm::dyn_cast<omp::ParallelOp>(owner)) {
      if (llvm::is_contained(parallelOp.getPrivateVars(), val)) {
        return true;
      }
    } else if (auto wsloopOp = llvm::dyn_cast<omp::WsloopOp>(owner)) {
      if (llvm::is_contained(wsloopOp.getPrivateVars(), val)) {
        return true;
      }
    } else if (auto simdOp = llvm::dyn_cast<omp::SimdOp>(owner)) {
      if (llvm::is_contained(simdOp.getPrivateVars(), val)) {
        return true;
      }
    }
  }
  return false;
}

template<typename T>
static void shadowGlobalVarWithLocalVar(T wrapper, Value globalVar, OpBuilder opBuilder) {
  auto privateVars = wrapper.getPrivateVars();
  auto it = llvm::find(privateVars, globalVar);
  if (it != privateVars.end()) {
    // showdow global var with local var
    unsigned idx = std::distance(privateVars.begin(), it);
    auto innerArg = wrapper.getRegion().front().getArgument(idx);

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
      return wrapper->isAncestor(globalUse.getOwner());
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
    wrapper.getPrivateVarsMutable().erase(idx);
    if (auto syms = wrapper.getPrivateSyms()) {
      llvm::SmallVector<mlir::Attribute> newSyms(syms->getValue());
      newSyms.erase(newSyms.begin() + idx);
      if (newSyms.empty()) {
        wrapper.removePrivateSymsAttr();
    } else {
        wrapper.setPrivateSymsAttr(opBuilder.getArrayAttr(newSyms));
      }
    }
    wrapper.getRegion().front().eraseArgument(idx);

    opBuilder.restoreInsertionPoint(savedPtr);
  }
  return; 
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
static void dropAllUsesOfDeclareOpRes(
  const llvm::SmallVector<Operation*>& wrappers, 
  omp::TargetOp targetOp, 
  hlfir::DeclareOp dOp, 
  OpBuilder& opBuilder
) {
  auto res0 = dOp.getResult(0);
  auto res1 = dOp.getResult(1);
  for (size_t i = 0; i < wrappers.size(); i++) {
    if (auto teamsOp = llvm::dyn_cast<omp::TeamsOp>(wrappers[i])) {
      shadowGlobalVarWithLocalVar(teamsOp, res0, opBuilder);
      shadowGlobalVarWithLocalVar(teamsOp, res1, opBuilder);
    } else if (auto parallelOp = llvm::dyn_cast<omp::ParallelOp>(wrappers[i])) {
      shadowGlobalVarWithLocalVar(parallelOp, res0, opBuilder);
      shadowGlobalVarWithLocalVar(parallelOp, res1, opBuilder);
    } else if (auto wsloopOp = llvm::dyn_cast<omp::WsloopOp>(wrappers[i])) {
      shadowGlobalVarWithLocalVar(wsloopOp, res0, opBuilder);
      shadowGlobalVarWithLocalVar(wsloopOp, res1, opBuilder);
    } else if (auto simdOp = llvm::dyn_cast<omp::SimdOp>(wrappers[i])) {
      shadowGlobalVarWithLocalVar(simdOp, res0, opBuilder);
      shadowGlobalVarWithLocalVar(simdOp, res1, opBuilder);
    }
  }
  return;
}

static void replaceWithFirDoLoop(
  omp::LoopNestOp lNOp,
  OpBuilder& opBuilder
) {
  auto saved = opBuilder.saveInsertionPoint();

  fir::DoLoopOp outmostLoopOp = nullptr;
  fir::DoLoopOp lastLoopOp = nullptr;
  llvm::SmallVector<mlir::Value> newInductionVars;
  // from outmost to innermost
  for (uint64_t i = 0; i < lNOp.getCollapseNumLoops(); i ++) {
    auto lb = lNOp.getLoopLowerBounds()[i];
    auto ub = lNOp.getLoopUpperBounds()[i];
    auto step = lNOp.getLoopSteps()[i];
    if (!outmostLoopOp) {
      opBuilder.setInsertionPoint(lNOp);
    } else {
      opBuilder.setInsertionPointToStart(lastLoopOp.getBody());
    }
    if (!lb.getType().isIndex()) {
      auto lbConvertOp = fir::ConvertOp::create(opBuilder, lNOp.getLoc(), IndexType::get(opBuilder.getContext()), lb, {});
      lb = lbConvertOp.getResult();
    }
    if (!ub.getType().isIndex()) {
      auto ubConvertOp = fir::ConvertOp::create(opBuilder, lNOp.getLoc(), IndexType::get(opBuilder.getContext()), ub, {});
      ub = ubConvertOp.getResult();
    }
    if (!step.getType().isIndex()) {
      auto stepConvertOp = fir::ConvertOp::create(opBuilder, lNOp.getLoc(), IndexType::get(opBuilder.getContext()), step, {});
      step = stepConvertOp.getResult();
    }
    auto loopOp = fir::DoLoopOp::create(opBuilder, lNOp.getLoc(), lb, ub, step);
    if (outmostLoopOp == nullptr) {
      outmostLoopOp = loopOp;
    }
    lastLoopOp = loopOp;
    newInductionVars.push_back(loopOp.getInductionVar()); 
  }

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
    // sourceBlock->getArgument(i).replaceAllUsesWith(newInductionVars[i]);
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


static void flattenTargetOp(
  llvm::SmallVector<Operation*>& wrappers, 
  omp::TargetOp targetOp, 
  OpBuilder& builder
) {
  llvm::SmallVector<omp::LoopNestOp> loopNests;
  targetOp.walk([&](omp::LoopNestOp lNOp) {
    loopNests.push_back(lNOp);
  });

  for (auto lNOp : loopNests) {
    replaceWithFirDoLoop(lNOp, builder);
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
    auto* terminator = innerBlock->getTerminator();
    auto insertEnd = innerOps.end();
    if (llvm::isa_and_nonnull<omp::TerminatorOp>(terminator)) {
      insertEnd = mlir::Block::iterator(terminator);
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
        auto &region = targetOp.getRegion();
        auto &block = region.front();
        auto mapVars = targetOp.getMapVarsMutable();
        auto mapVarsMutable = targetOp.getMapVarsMutable();
        int mapEntrySize = mapVarsMutable.size();

        llvm::SmallVector<Operation*> shadowedDecalreOps;
        llvm::SmallVector<Operation*> wrappers = getOmpWrappers(targetOp);
        for (int i = mapEntrySize - 1; i >= 0; i--) {
          auto arg = block.getArgument(i);
          if (llvm::range_size(arg.getUsers()) != 1) {
            continue;
          }
          Operation* userOp = *arg.getUsers().begin();
          auto dOp = llvm::dyn_cast<hlfir::DeclareOp>(userOp);
          if (!dOp) {continue;}
          assert(dOp.getNumResults() == 2);
          auto declaredOperand = dOp.getResult(0);
          if (!inPrivateVars(declaredOperand)) {
            continue;
          }

          dropAllUsesOfDeclareOpRes(wrappers, targetOp, dOp, opBuilder);
          dOp.erase();
          arg.dropAllUses();
          block.eraseArgument(i);
          mapVarsMutable.erase(i);
        }
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
