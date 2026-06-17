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
#include <clang/Basic/DiagnosticSema.h>
#include <clang/Parse/Parser.h>
#include <llvm/Support/DebugLog.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsAttributes.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
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
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <utility>
#include <variant>

namespace flangomp {
#define GEN_PASS_DEF_LOWERWORKDISTRIBUTETOJIT
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "lower-workdistribute-to-stablehlo"

using namespace mlir;

namespace {

static unsigned getBlockHostEvalVarsOffset(omp::TargetOp targetOp) {
  return targetOp.numHasDeviceAddrBlockArgs();
}

static unsigned getBlockMapVarsOffset(omp::TargetOp targetOp) {
  return targetOp.numHasDeviceAddrBlockArgs()
    + targetOp.numInReductionBlockArgs()
    + targetOp.numHostEvalBlockArgs();
}


// TODO: this might not be necessary as we only what to know if it is defined by an alloca or not?
// Originally I think I need to splice all the track to the definition, but actually I can just shadow this.
static std::optional<llvm::SmallVector<Operation*>> getSortedDefinitionTrack(omp::TargetOp targetOp, const Value& candidate) {
  llvm::SmallVector<Operation*> track;

  auto subroutineFunc = targetOp->getParentOfType<func::FuncOp>();
  auto curr = candidate;
  auto* definedOp = curr.getDefiningOp();
  track.push_back(definedOp);
  auto shouldEnd = [&](Operation* definedOp) -> bool {
    if (!definedOp) return true; // also include the case when val is in block
    return llvm::isa<fir::AllocaOp>(definedOp) || !subroutineFunc->isAncestor(definedOp);
  };

  while (!shouldEnd(definedOp)) {
    if (auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(definedOp)) {
      curr = mapInfoOp.getVarPtr();
      definedOp = curr.getDefiningOp(); 
      track.push_back(definedOp);
      continue;
    } 
    if (auto viewOp = llvm::dyn_cast<mlir::ViewLikeOpInterface>(definedOp)) {
      curr = viewOp.getViewSource();
      definedOp = curr.getDefiningOp();
      track.push_back(definedOp);
      continue;
    }
    if (auto hlfirDeclareOp = llvm::dyn_cast<hlfir::DeclareOp>(definedOp)) {
      curr = hlfirDeclareOp.getMemref();
      definedOp = curr.getDefiningOp();
      track.push_back(definedOp);
      continue;
    }
    // unexepected situation, should 
    llvm::dbgs() << "Unexpected Situation: ";
    definedOp->print(llvm::dbgs());
    llvm::dbgs() << "\n";
    track.push_back(nullptr);
    break;
  }
  if (!track[track.size() - 1]) {return std::nullopt;}
  std::reverse(track.begin(), track.end());
  return track;
}

struct AliasVar {
  llvm::DenseSet<Value> aliasSet;
  Value currentValue;
};

struct ScanResult {
  llvm::SmallVector<std::unique_ptr<AliasVar>> aliasVarStorage;
  llvm::DenseMap<Value, AliasVar*> memToAliasVar;
  llvm::DenseMap<Value, std::pair<Value, Operation*>> dataDAG;

  void print() {
    llvm::dbgs() << "\ndataDAG:";
    for (auto entry: dataDAG) {
      llvm::dbgs() << "\n"; 
      entry.getFirst().printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << ": ";
      if (entry.getSecond().first) {
        entry.getSecond().first.printAsOperand(llvm::dbgs(), {});
      } else {
        entry.getSecond().second -> print(llvm::dbgs());
      }
    }
    llvm::dbgs() << "\nend data DAG\n";

    llvm::dbgs() << "\nAliasVars:";
    for (const auto& aliasVar : aliasVarStorage) {
      llvm::dbgs() << "\n(";
      for (const auto& entry: aliasVar->aliasSet) {
        entry.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << ", ";
      }
      llvm::dbgs() << "):\n";
      if (aliasVar -> currentValue) {
        aliasVar->currentValue.printAsOperand(llvm::dbgs(), {});
      } else {
        llvm::dbgs() << "nullptr";
      }
    }
    llvm::dbgs() << "\nEnd AliasVars\n";
  }

  // e.g. mem = fir.alloca
  void newMem(Value mem) {
    assert(!memToAliasVar.contains(mem));
    auto aliasVar = std::make_unique<AliasVar>();
    aliasVar->aliasSet.insert(mem);
    aliasVar->currentValue = nullptr;

    memToAliasVar[mem] = aliasVar.get();
    aliasVarStorage.push_back(std::move(aliasVar));
    return;
  }

  // e.g. fir.store val to mem, hlfir.assign val to mem
  void assignValueToMem(Value mem, Value val) {
    if (!memToAliasVar.contains(mem)) {
      newMem(mem);
    }
    assert(memToAliasVar.contains(mem));
    memToAliasVar[mem]->currentValue = val;
    return;
  };

  // e.g. newMem = hlfir.declare(oldMem)
  // e.g. memCopiedTo = map.info var_ptr(memCopiedFrom) map_clauses(implicit ToFrom) capture(ByRef)
  void bindMems(Value oldMem, Value newMem) {
    assert(memToAliasVar.contains(oldMem));
    auto* aV = memToAliasVar[oldMem];
    aV->aliasSet.insert(newMem);
    memToAliasVar[newMem] = aV;
    return;
  }

  // e.g. memCopiedTo = map.info var_ptr(memCopiedFrom) map_clauses(implicit) capture(ByCopy)
  void copyMem(Value memCopiedFrom, Value memCopiedTo) {
    newMem(memCopiedTo);
    assert(memToAliasVar.contains(memCopiedTo));
    assert(memToAliasVar.contains(memCopiedFrom));
    memToAliasVar[memCopiedTo]->currentValue = memToAliasVar[memCopiedFrom]->currentValue;
    return;
  }

  void loadValue(Value val, Value memLoadFrom) {
    assert(memToAliasVar.contains(memLoadFrom));
    assert(!dataDAG.contains(val));
    dataDAG[val] = std::make_pair(memToAliasVar[memLoadFrom]->currentValue, nullptr); 
    return;
  }

  void flowValue(Value val, Operation* op) {
    assert(!dataDAG.contains(val));
    dataDAG[val] = std::make_pair(nullptr, op);
    return;
  }
};

static ScanResult scanValuesUntilTarget(omp::TargetOp targetOp) {
  ScanResult res;

  auto funcOp = targetOp->getParentOfType<func::FuncOp>();
  assert(targetOp->getBlock() == &funcOp.getRegion().front());
  auto& entryBlock = funcOp.getRegion().front();
  for (auto arg : entryBlock.getArguments()) {
    res.assignValueToMem(arg, arg);
  }

  Operation& op = entryBlock.getOperations().front();
  Operation* opPtr = &op;
  while (opPtr != targetOp) {
    // TODO: consider the case when we have multiple targetOp, we should be able to skip the previous one!
    if (opPtr->getNumRegions() > 0) {
      llvm::errs() << "Unexpected Operation!\n";
      std::exit(EXIT_FAILURE);
    }
    llvm::TypeSwitch<Operation*>(opPtr)
      .Case([&](fir::AllocaOp allocaOp){res.newMem(allocaOp.getResult());})
      .Case([&](fir::StoreOp storeOp){res.assignValueToMem(storeOp.getMemref(), storeOp.getValue());})
      .Case([&](hlfir::AssignOp assignOp){res.assignValueToMem(assignOp.getLhs(), assignOp.getRhs());})
      // TODO: need to check the case of slice!
      .Case([&](hlfir::DeclareOp declareOp){
        assert(declareOp.getNumResults() == 2);
        res.bindMems(declareOp.getMemref(), declareOp.getResult(0));
        res.bindMems(declareOp.getMemref(), declareOp.getResult(1));
      })
      .Case([&](fir::ConvertOp convertOp){
        // TODO: check this
        if (llvm::isa<fir::ReferenceType>(convertOp.getResult().getType())) {
          res.bindMems(convertOp.getOperand(), convertOp.getResult());
        } else {
          res.flowValue(convertOp.getResult(), convertOp);
        }
      })
      .Case([&](fir::LoadOp loadOp){res.loadValue(loadOp.getResult(), loadOp.getMemref());})
      .Case([&](fir::AddrOfOp addrOp){res.newMem(addrOp.getResult());}) // TODO: in fact I should just ignore this, but I also need to trim the whole chain
      // %303 = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
      .Case([&](omp::MapInfoOp mapInfoOp){
        if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
          assert((uint32_t(mapInfoOp.getMapType()) & uint32_t(omp::ClauseMapFlags::from)) == 0); // NOTE:maybe not necessary
          res.copyMem(mapInfoOp.getVarPtr(), mapInfoOp.getResult());
        } else if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByRef) {
          assert((uint32_t(mapInfoOp.getMapType()) & uint32_t(omp::ClauseMapFlags::from)) != 0); // NOTE:maybe not necessary
          res.bindMems(mapInfoOp.getVarPtr(), mapInfoOp.getResult());
        } else {
          llvm::errs() << "Unexpected operation!";
          mapInfoOp->print(llvm::errs());
          llvm::errs() << "\n";
          std::exit(EXIT_FAILURE);
        }
      })
      .Default([&](auto){
        // arith, math, complex
        if (mlir::isPure(opPtr) && opPtr->getNumResults() == 1) {
          res.flowValue(opPtr->getResult(0), opPtr);
        } else {
          llvm::dbgs() << "Skipped operation:";
          opPtr->print(llvm::dbgs());
          llvm::dbgs() << "\n";
        }
      })
    ;
    opPtr = opPtr->getNextNode();
  }
  return res;
}

static bool dependOnArg(
  const Value& val,
  const llvm::DenseMap<Value, std::pair<Value, Operation*>>& dataDAG,
  const llvm::DenseSet<Value>& argsSet
) {
  if (argsSet.contains(val)) return true;
  assert(dataDAG.contains(val)); 
  auto pair = dataDAG.at(val);
  assert(!(pair.first && pair.second));
  if (pair.first) {
    return dependOnArg(pair.first, dataDAG, argsSet);
  } 
  assert(pair.second);
  return llvm::any_of(pair.second->getOperands(), [&](const auto& operand) -> bool{return dependOnArg(operand, dataDAG, argsSet);});
}

// There are 3 possible situations for private and first private variables.
// (1) Private locally allocated purely for storing temporary variables.
// (2) First Private assigned with constant value before target region.
// (3) First Private assigned depend on input arguments (runtime constants though).
// Fold (1), (2); But not (3).
static bool shouldFold(
  const Value& candidate,
  const ScanResult& sr, 
  func::FuncOp funcOp,
  const llvm::SmallVector<Operation*> &definitionTrack,
  const llvm::DenseSet<Value>& argsSet
) {
  auto* declaredOp = definitionTrack.empty() ? nullptr : definitionTrack.back();
  auto isTargetOrPointer = [](Operation* definedOp) {
    return false; // TODO: finish this method.
  };
  if (!declaredOp || !funcOp->isAncestor(declaredOp) || isTargetOrPointer(declaredOp)) {
    return false;
  }

  bool hasDeclared = false; // This is to avoid folding shape metas.
  bool isLocallyAllocated = false;
  for (auto* op: definitionTrack) {
    if (llvm::isa<hlfir::DeclareOp>(op)) {
      hasDeclared = true;
    } else if (llvm::isa<fir::AllocaOp>(op)) {
      isLocallyAllocated = true;
    }
    if (hasDeclared && isLocallyAllocated) break;
  }
  if (!(hasDeclared && isLocallyAllocated)) return false;

  assert(sr.memToAliasVar.contains(candidate)); // Should not skip any mem during the scan.
  auto* aV = sr.memToAliasVar.at(candidate);
  if (aV->currentValue == nullptr) return true; // This belongs to case (1).
  return !dependOnArg(aV->currentValue, sr.dataDAG, argsSet);
}

// Print the dataflow chain of val to the target location
static Value materializeValue(
  Value val, 
  const llvm::DenseMap<Value, std::pair<Value, Operation*>>& dataDAG,
  llvm::DenseMap<Value, Value>& cloneCache,
  OpBuilder& opBuilder
) {
  assert(dataDAG.contains(val));
  if (cloneCache.contains(val)) {
    return cloneCache[val];
  }
  auto [upVal, defOp] = dataDAG.at(val);

  // If only there is a value, transformed from a load operation.
  assert(!(upVal && defOp));
  assert(upVal || defOp);
  if (upVal) {
    auto result = materializeValue(upVal, dataDAG, cloneCache, opBuilder);
    cloneCache[val] = result;
    return result;
  }

  // If has a corresponding operation.
  IRMapping vMap;
  for (auto operand: defOp->getOperands()) {
    vMap.map(operand, materializeValue(operand, dataDAG, cloneCache, opBuilder));
  }
  auto* newOp = opBuilder.clone(*defOp, vMap);
  cloneCache[val] = newOp->getResult(0);
  return cloneCache[val];
}

static Value createLocalAlloca(
  const llvm::SmallVector<Operation*>& definitionTrack,
  Block& entryBlock,
  Value arg,
  OpBuilder& opBuilder,
  omp::TargetOp targetOp
) {
  auto oldAllocaOp = llvm::dyn_cast<fir::AllocaOp>(definitionTrack.front());
  assert(oldAllocaOp);
  
  auto newAllocaOp = fir::AllocaOp::create(
    opBuilder, 
    targetOp.getLoc(), 
    oldAllocaOp.getInType(), 
    oldAllocaOp.getUniqName().has_value()? oldAllocaOp.getUniqName().value(): "",
    oldAllocaOp.getBindcName().has_value()? oldAllocaOp.getBindcName().value(): "",
    oldAllocaOp.getTypeparams(),
    oldAllocaOp.getShape(),
    oldAllocaOp->getAttrs()
  );

  mlir::Value replacementVal = newAllocaOp.getResult();
  if (replacementVal.getType() != arg.getType()) {
    replacementVal = fir::ConvertOp::create(opBuilder, targetOp.getLoc(), arg.getType(), replacementVal).getResult();
  }
  return replacementVal;
}

static void assignToLocalAlloca(
  Value replacementVal,
  omp::TargetOp targetOp,
  int mapVarIdx,
  const ScanResult& scanResult,
  OpBuilder& opBuilder,
  llvm::DenseMap<Value, Value>& cloneCache
) {
  auto mapVar = targetOp.getMapVars()[mapVarIdx];
  assert(scanResult.memToAliasVar.contains(mapVar));
  llvm::dbgs() << "Materialize the value for mapVar with index " << mapVarIdx << ", var: ";
  mapVar.printAsOperand(llvm::dbgs(), {});
  llvm::dbgs() << " , mapVarVal: ";
  auto mapVarVal = scanResult.memToAliasVar.at(mapVar)->currentValue;
  if (mapVarVal != nullptr) {
    mapVarVal.printAsOperand(llvm::dbgs(), {});
    auto finalVal = materializeValue(mapVarVal, scanResult.dataDAG, cloneCache, opBuilder);
    hlfir::AssignOp::create(opBuilder, targetOp.getLoc(), finalVal, replacementVal);
  } else {
    auto zeroConstant = arith::getZeroConstant(opBuilder, targetOp.getLoc(), fir::unwrapRefType(replacementVal.getType()));
    hlfir::AssignOp::create(opBuilder, targetOp.getLoc(), zeroConstant, replacementVal);
  }
  llvm::dbgs() << " .\n";
}

static void handleTemporaryVariables(omp::TargetOp targetOp, OpBuilder& opBuilder) {
  auto funcOp = targetOp->getParentOfType<func::FuncOp>();
  auto& entryBlock = targetOp.getRegion().front(); 
  auto funcArgs = funcOp.getRegion().front().getArguments(); 
  auto funcArgsSet = llvm::DenseSet<Value>(funcArgs.begin(), funcArgs.end());
  auto scanResult = scanValuesUntilTarget(targetOp);
  // scanResult.print();
  auto mapVarOffset = getBlockMapVarsOffset(targetOp);
  assert(mapVarOffset == 0); // as other argumemts have already been folded inside map_entries

  OpBuilder::InsertionGuard guard(opBuilder); 
  opBuilder.setInsertionPointToStart(&entryBlock);
  llvm::SmallVector<int> indicesToFold;
  llvm::DenseMap<Value, Value> cloneCache;
  for (int i = targetOp.getMapVars().size() - 1; i >= 0; i--) {
    auto candidate = targetOp.getMapVars()[i];
    auto definitionTrackOpt = getSortedDefinitionTrack(targetOp, candidate);
    if (!definitionTrackOpt.has_value()) continue;
    auto definitionTrack = definitionTrackOpt.value();
    if (!shouldFold(candidate, scanResult, funcOp, definitionTrack, funcArgsSet)) continue;
    assert(llvm::isa<fir::AllocaOp>(definitionTrack.front()));

    indicesToFold.push_back(i);
    auto argIdx = mapVarOffset + i; 
    auto arg = entryBlock.getArgument(argIdx);
    auto replacementVal = createLocalAlloca(definitionTrack, entryBlock, arg, opBuilder, targetOp);
    assignToLocalAlloca(replacementVal, targetOp, i, scanResult, opBuilder, cloneCache);
    arg.replaceAllUsesWith(replacementVal);
    assert(arg.getUses().empty());
    
    // Clear the arguments
    assert(entryBlock.getArgument(argIdx).getUses().empty());
    entryBlock.eraseArgument(argIdx);
    targetOp.getMapVarsMutable().erase(i);
  }
  return;
}


using PrivateOmpOp = std::variant<omp::TeamsOp, 
                                  omp::ParallelOp, 
                                  omp::DistributeOp,
                                  omp::WsloopOp, 
                                  omp::SimdOp>;


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
    .Case([&](omp::SimdOp simdOp){
      // [private_vars, reduction_vars]
      offset = simdOp.numPrivateBlockArgs();
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

static void shadowGlobalVarWithLocalVar(PrivateOmpOp wrapper, Value globalVar, OpBuilder& opBuilder) {
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
    for (int i = promotedArgs.size() - 1; i >= 0; i--) {
      if (promotedArgs[i] == rmIdx) {
        promotedArgs.erase(promotedArgs.begin() + i);
      } else if (promotedArgs[i] > rmIdx) {
        promotedArgs[i] -= 1;
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
  assert(wrapper.getReductionVars().size() == size_t(reductionVarsCount));


  llvm::DenseSet<Operation*> toErase;
  for (int i = 0; i < reductionVarsCount; i++) {
    Value reductionVarArg = entryBlock.getArgument(reductionVarsOffset + i);
    Value reductionVar = wrapper.getReductionVars()[i];

    for (auto* user: reductionVarArg.getUsers()) {
      if (auto declaredOp = llvm::dyn_cast<hlfir::DeclareOp>(user)) {
        assert(declaredOp.getMemref() == reductionVarArg);
        declaredOp.getResult(0).replaceAllUsesWith(reductionVar);
        if (declaredOp.getNumResults() > 1) {
          assert(declaredOp.getNumResults() == 2);
          assert(declaredOp.getResult(1).use_empty());
        }
        assert(declaredOp.use_empty());
        toErase.insert(declaredOp);
      }    
    }
    reductionVarArg.replaceAllUsesWith(reductionVar);
    assert(reductionVarArg.use_empty());
  }

  llvm::for_each(toErase, [](Operation* op){op->erase();});

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
      // Should in an inner -> outer ordering!
      applyToConcreteType<omp::SimdOp>(wrapper, replaceLocalReductionVars<omp::SimdOp>)
      || applyToConcreteType<omp::WsloopOp>(wrapper, replaceLocalReductionVars<omp::WsloopOp>)
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
      if (targetOp->walk([](omp::WorkdistributeOp) { return WalkResult::interrupt();}).wasInterrupted()) {
        // FIXME: unify the logic with the next branch
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
        // llvm::dbgs() << "\nA1:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA1 End\n";
        llvm::SmallVector<int> indicesForAbsorbedHostEvalVars;
        absorbHostEvalVarsToMapEntries(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        removePrivateVarsFromMapEntry(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        // llvm::dbgs() << "\nA2:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA2 End\n";
        auto wrappers = getOmpWrappers(targetOp);
        flattenTargetOp(wrappers, targetOp, opBuilder);
        // llvm::dbgs() << "\nA3:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA3 End\n";

        PassManager pm(&context);
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (failed(pm.run(moduleOp))) {
          llvm::errs() << "Fail to preprocess the moduleOp!\n";
          // TODO: should have fallback processing!
          std::exit(EXIT_FAILURE);
        };
        // llvm::dbgs() << "\nA4:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA4 End\n";
        handleTemporaryVariables(targetOp, opBuilder);
        // llvm::dbgs() << "\nA5:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA5 End\n";
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
