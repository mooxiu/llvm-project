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

#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsAttributes.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <string>
#include <strstream>
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


struct SubState {
  enum class State {Bottom, Known, Top};
  State state = State::Bottom;
  Value value = nullptr;

  bool operator==(const SubState &rhs) const {
    if (state != rhs.state) return false;
    if (state == State::Known) {
      return value == rhs.value;
    }
    return true;
  }

  bool operator!=(const SubState &rhs) const {
    return !(*this == rhs); 
  }

  // Usecase example: `omp target exit data map(delete)`
  void reset() {
    this->state = State::Bottom;
    this->value = nullptr;
  }

  static SubState join(
    const SubState& lhs, 
    const SubState& rhs 
  ) {
    if (lhs.state == State::Top || rhs.state == State::Top) return {State::Top, nullptr};
    if (lhs.state == State::Bottom) return rhs;
    if (rhs.state == State::Bottom) return lhs;
    assert(lhs.state == State::Known && rhs.state == State::Known);
    if (lhs.value == rhs.value) return lhs;
    return {State::Top, nullptr};
  }

  std::string toStr() const {
    auto valueToString = [](mlir::Value value) -> std::string {
      if (!value) return "NULL";
      std::string str;
      llvm::raw_string_ostream os(str);
      value.printAsOperand(os, mlir::OpPrintingFlags{});
      os.flush();
      return str;
    };

    std::string stateStr;
    std::string valueStr;
    if (this->state == State::Bottom) {
      stateStr = "BOT";
      valueStr = "NULL";
    } else if (this->state == State::Known) {
      stateStr = "KNO";
      valueStr = valueToString(this->value);
    } else {
      stateStr = "TOP"; 
      valueStr = "NULL";
    }
    return llvm::formatv("({0}, {1})", stateStr, valueStr);
  }
};

// Definition of state lattice.
// Bottom: Minimum uncertainty, unitialized, unreacheable, etc..
// Known: Known value.
// Top: Maximum uncertainty, unknown, for example, the join state of a value after assigned in an if block.
struct MemState {
  SubState hostSubState = {SubState::State::Bottom, nullptr};
  SubState deviceSubState = {SubState::State::Bottom, nullptr};
  std::pair<int, int> refCountRange = {0, 0}; 

  bool operator==(const MemState &rhs) const {
    return (hostSubState == rhs.hostSubState)
      && (deviceSubState == rhs.deviceSubState)
      && (refCountRange == rhs.refCountRange);
  }

  bool operator!=(const MemState &rhs) const {
    return !(*this == rhs);
  }

  static MemState join(
    const MemState& lhs, 
    const MemState& rhs 
  ) {
    MemState ms;
    ms.hostSubState = SubState::join(lhs.hostSubState, rhs.hostSubState);
    ms.deviceSubState = SubState::join(lhs.deviceSubState, rhs.deviceSubState);
    ms.refCountRange = {std::min(lhs.refCountRange.first, rhs.refCountRange.first), std::max(lhs.refCountRange.second, rhs.refCountRange.second)};
    return ms;
  }

  int incRefCount(int count) {
    if (count < INT_MAX) return count + 1;
    return count;
  } 

  int decRefCount(int count) {
    return std::max(int(0), count - 1);
  }

  void incRefCounts() {
    this->refCountRange = {
      incRefCount(this->refCountRange.first), 
      incRefCount(this->refCountRange.second)
    };
    return;
  }

  void decRefCounts() {
    this->refCountRange = {
      decRefCount(this->refCountRange.first), 
      decRefCount(this->refCountRange.second)
    };
    return;
  }

  std::string toStr() const {
    return llvm::formatv(
      "({0}, {1}, ({2}, {3})}", 
      this->hostSubState.toStr(), 
      this->deviceSubState.toStr(), 
      this->refCountRange.first,
      this->refCountRange.second
    );
  }
};

// Instead of maintaing an alias set.
// TODO: think if slicing will also fit here.
static Value getRootMem(Value mem) {
  llvm::dbgs() << "\nfinding root of: ";
  Value curr = mem;
  while (Operation* defOp = curr.getDefiningOp()) {
    curr.printAsOperand(llvm::dbgs(), {});
    llvm::dbgs() << ", ";
    if (auto declare = llvm::dyn_cast<hlfir::DeclareOp>(defOp)){
      curr = declare.getMemref();
      continue;
    } 
    if (auto declare = llvm::dyn_cast<fir::DeclareOp>(defOp)) { // INFO: in fact, should not exist here.
      curr = declare.getMemref();
      continue;
    }
    if (auto convOp = llvm::dyn_cast<fir::ConvertOp>(defOp)) {
      curr = convOp.getValue();
      continue;
    } 
    if (auto allocaOp = llvm::dyn_cast<fir::AllocaOp>(defOp)) {
      return curr;
    }
    if (auto addrOp = llvm::dyn_cast<fir::AddrOfOp>(defOp)) {
      return curr;
    }
    if (auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(defOp)) {
      if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
        return curr;
      } 
      if (mapInfoOp.getMapCaptureType() == omp::VariableCaptureKind::ByRef) {
        curr = mapInfoOp.getVarPtr();
        continue;
      } 
    }
    // TODO: maybe we need to distinguish the specific indices to make finer grainer analysis.
    if (auto designateOp = llvm::dyn_cast<hlfir::DesignateOp>(defOp)) {
      curr = designateOp.getMemref(); // conservative
      continue;
    }
    if (auto boxAddr = llvm::dyn_cast<fir::BoxAddrOp>(defOp)) {
      curr = boxAddr.getVal();
      continue;
    }
    if (auto boxOffset = llvm::dyn_cast<fir::BoxOffsetOp>(defOp)) {
      curr = boxOffset.getBoxRef();
      continue;
    }
    // INFO: the result of load can also be a mem:
    // %356 = fir.load %355 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>
    if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(defOp)) {
      curr = loadOp.getMemref();
      continue;
    }
    llvm::errs() << "\n Unexpected defining operation: ";
    defOp->print(llvm::errs());
    llvm::errs() << "\n";
    std::exit(EXIT_FAILURE);
  }
  return curr;
}

class MemoryAliasLattice: public mlir::dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemoryAliasLattice)

  using dataflow::AbstractDenseLattice::AbstractDenseLattice;

  // Only record the root memory's MemoryValueState, so we need to use getRootMem function to get the root first.
  llvm::DenseMap<Value, MemState> memoryMap;

  ChangeResult join(const mlir::dataflow::AbstractDenseLattice &rhs) override {
    auto changed = false; 
    const auto& rhsLattice = static_cast<const MemoryAliasLattice&>(rhs);
    for (const auto& [mem, state] : rhsLattice.memoryMap) {
      assert(mem == getRootMem(mem));
      if (!memoryMap.contains(mem)) changed = true;
      auto lhsState = memoryMap[mem]; // auto create bottom if not exist 
      auto rhsState = rhsLattice.memoryMap.at(mem);
      auto joinState = MemState::join(lhsState, rhsState);
      if (lhsState != joinState) changed = true;
      memoryMap[mem] = joinState;
    }
    return changed? mlir::ChangeResult::Change : mlir::ChangeResult::NoChange;
  }

  void print(llvm::raw_ostream &os) const override {
    os << "\n[DEBUG] Current MemoryMap: ";
    for (const auto& entry: memoryMap) {
      os << "\n (Mem: "; 
      entry.getFirst().printAsOperand(os, {});
      os << ", State: " << entry.getSecond().toStr() << ")";
    }
  };
};

class TargetFoldabilityAnalysis: public mlir::dataflow::DenseForwardDataFlowAnalysis<MemoryAliasLattice> {
private:
  func::FuncOp currentFunc;

  void opDataUpdate(
    ::mlir::Operation::operand_range mapVars, 
    MemoryAliasLattice* after,
    llvm::function_ref<void(MemState&, ::mlir::omp::ClauseMapFlags, ::mlir::omp::VariableCaptureKind)> updater
  ) {
    for (const auto& mapVar: mapVars) {
      auto root = getRootMem(mapVar);
      llvm::dbgs() << "\n the mapVar is: ";
      mapVar.printAsOperand(llvm::dbgs(), {});
      llvm::dbgs() << ", the root is: ";
      root.printAsOperand(llvm::dbgs(), {});
      // assert(after->memoryMap.contains(root)); FIXME: think a better way of handling this
      auto& state = after->memoryMap[root];
      assert(llvm::isa<omp::MapInfoOp>(mapVar.getDefiningOp()));
      auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(mapVar.getDefiningOp()); 
      assert(mapInfoOp);
      auto direction = mapInfoOp.getMapType();
      auto captureType = mapInfoOp.getMapCaptureType();
      updater(state, direction, captureType);
    }
  } 

  // These 3 are standalone operations
  void handleTargetUpdateOp(omp::TargetUpdateOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);
      assert(!(hasFrom && hasTo));
      if (hasFrom) {
        // from device to host
        state.hostSubState = state.deviceSubState;
        // refCount no change
        return;
      }
      if (hasTo) {
        // from host to device 
        state.deviceSubState = state.hostSubState;
        // refCount no change
        return;
      } 
      llvm_unreachable("Unexpected target update direction");
      return;
    });
  };

  void handleTargetEnterDataOp(omp::TargetEnterDataOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      // According to https://www.openmp.org/spec-html/5.0/openmpsu58.html: A map-type must be specified in all map clauses and must be either to or alloc.
      // `alloc` will be come `storage` clause
      bool hasAlloc = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);
      // common clause flag
      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      assert(hasAlloc || hasTo);
      assert(!(hasAlloc && hasTo));
      if (hasAlloc) {
        // 1. If not mapped (0, 0), alloc undefined
        // 2. If mapped (Z+, Z+), keep the same
        // 3. If (0, Z+), then JOIN(deviceValue, TOP) = TOP, the same with 1
        if (state.refCountRange.first == 0) {
          state.deviceSubState = {SubState::State::Top, nullptr};
        } else {
          // DO NOTHING
        }
        state.incRefCounts();
        return;
      } 
      if (hasTo) {
        state.incRefCounts();
        if (hasAlways || state.refCountRange.second == 0) {
          state.deviceSubState = state.hostSubState;
          return;
        } 
        // {*, Z+}
        if (state.refCountRange.first == 0) {
          // {0, Z+}
          state.deviceSubState = SubState::join(state.hostSubState, state.deviceSubState);
          return;
        } 
        // {Z+, Z+}
        // DO NOTHING except for refCountRange increase
        return;
      }
      llvm_unreachable("Unexpected target update direction");
      return;
    });
  };

  void handleTargetExitDataOp(omp::TargetExitDataOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      //  A map-type must be specified in all map clauses and must be either from, release, or delete. (https://www.openmp.org/spec-html/5.0/openmpsu59.html)
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);  
      bool hasRelease = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);
      bool hasDelete = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::del);
      assert((int)hasFrom + (int)hasRelease + (int)hasDelete == 1 && "target exit data map-type must be from, release, or delete");
      // common flag
      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      if (hasFrom) {
        if (hasAlways) {
          // (0, 0) do nothing
          // (0, Z+) host = join(host, device) 
          // (1, Z+) set host to device
          if (state.refCountRange.first == 0) {
            if (state.refCountRange.second >= 1) {
              state.hostSubState = SubState::join(state.hostSubState, state.deviceSubState);
            }
          } else {
            // (1, Z+)
            state.hostSubState = state.deviceSubState;
          }
          state.decRefCounts();
          if (state.refCountRange.second == 0) {
            state.deviceSubState.reset();
          }
          return;
        }
        if (state.refCountRange.first == 0) {
          if (state.refCountRange.second == 0) {
            // (0, 0), DO NOTHING
          } else {
            // (0, Z+):
            // - if 0.. do nothing
            // - if 1.. copy back and reset
            // - if 1+.. do nothing
            state.hostSubState = SubState::join(state.hostSubState, state.deviceSubState);
            // device state = join(BOT, device state) = device state
          }
        } else if (state.refCountRange.first == 1){
          // - if 1: copy back and reset 
          // - else: do nothing
          if (state.refCountRange.second == 1) {
            state.hostSubState = state.deviceSubState;
          } else {
            state.hostSubState = SubState::join(state.hostSubState, state.deviceSubState);
          }
          // device state = join(BOT, device state) = device state
        } else {
          // DO NOTHING
        }
        state.decRefCounts();
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      if (hasRelease) {
        state.decRefCounts();
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      if (hasDelete) {
        state.deviceSubState.reset();
        state.refCountRange = {0, 0};
        return;
      }
      llvm_unreachable("Unexpected target update direction");
    });
  }

  void handleTargetOpEnteringEdge(omp::TargetOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      if (captureType == omp::VariableCaptureKind::ByCopy) {
        // If it is by copy, then it actually works like a first private. 
        return;
      }
      // Following are ByRef 
      assert(captureType == omp::VariableCaptureKind::ByRef);
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);  
      bool hasAlloc = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);

      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      assert(!(hasTo && hasAlloc));
      if (hasTo) {
        if (state.refCountRange.second == 0 || hasAlways) {
          // (0, 0)
          state.deviceSubState = state.hostSubState;
        } else {
          if (state.refCountRange.first == 0) {
            // (0, Z+)
            state.deviceSubState = SubState::join(state.deviceSubState, state.hostSubState);
          } else {
            // (Z+, Z+): DO NOTHING
          }
        }
        state.incRefCounts();
        return;
      }
      if (hasAlloc || (hasFrom && !hasTo)) {
        if (state.refCountRange.first == 0) {
          // (0, 0): device -> TOP
          // (0, Z+): device -> join(device, TOP) = TOP
          state.deviceSubState = {SubState::State::Top, nullptr};
        } else {
          // (Z+, Z+): do nothing
        }
        state.incRefCounts();
        return;
      }
      llvm_unreachable("ignore other cases for prototyping");
    });
    auto offSet = getBlockMapVarsOffset(op);
    for (unsigned int i = 0; i < op.numMapBlockArgs(); i++) {
      auto argIdx = i + offSet; 
      Value arg = op.getRegion().front().getArgument(argIdx);
      Value mapVar = op.getMapVars()[i];
      auto mapVarRoot = getRootMem(mapVar); 
      assert(after->memoryMap.contains(mapVarRoot));
      assert(llvm::isa<omp::MapInfoOp>(mapVar.getDefiningOp()));
      auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(mapVar.getDefiningOp());
      if (mapInfoOp.getMapCaptureType() == mlir::omp::VariableCaptureKind::ByCopy) {
        after->memoryMap[arg] = {
          after->memoryMap[mapVarRoot].hostSubState, 
          {SubState::State::Bottom, nullptr},
          {0, 0}
        };
      } else {
        after->memoryMap[arg] = {
          after->memoryMap[mapVarRoot].deviceSubState, 
          {SubState::State::Bottom, nullptr},
          {0, 0}
        };
      }
    }
  }

  void handleTargetOpExitingEdge(omp::TargetOp op, MemoryAliasLattice *after) {
    auto offSet = getBlockMapVarsOffset(op);
    for (unsigned int i = 0; i < op.numMapBlockArgs(); i++) {
      auto argIdx = i + offSet; 
      Value arg = op.getRegion().front().getArgument(argIdx);
      Value mapVar = op.getMapVars()[i];
      auto mapVarRoot = getRootMem(mapVar); 
      assert(llvm::isa<omp::MapInfoOp>(mapVar.getDefiningOp()));
      // FIXME: find a better way to handle this
      // assert(after->memoryMap.contains(mapVarRoot));
      // assert(after->memoryMap.contains(arg));
      auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(mapVar.getDefiningOp());
      if (mapInfoOp.getMapCaptureType() == mlir::omp::VariableCaptureKind::ByRef) {
        after->memoryMap[mapVarRoot].deviceSubState = after->memoryMap[arg].hostSubState;
      } else {
        // DO NOTHING if ByCopy
      }
    }
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      if (captureType == omp::VariableCaptureKind::ByCopy) {
        // If it is by copy, then it actually works like a first private. 
        return;
      }
      // Following are ByRef 
      assert(captureType == omp::VariableCaptureKind::ByRef);
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);  
      bool hasAlloc = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);

      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      if (hasFrom) {
        // conditional copy back
        // FIXME: find a way to assert
        // assert(state.refCountRange.first >= 1 && state.refCountRange.second >= 1);
        if (state.refCountRange.second == 1 || hasAlways) {
          // (1, 1) or always: copy back
          state.hostSubState = state.deviceSubState;
        } else {
          if (state.refCountRange.first == 1) {
            // (1, 1+)
            state.hostSubState = SubState::join(state.hostSubState, state.deviceSubState);
          } else {
            // (1+, 1+): do nothing
          }
        }
        state.decRefCounts();
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      if (hasAlloc || (hasTo && !hasFrom)) {
        // DO NOTHING
        state.decRefCounts(); 
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      llvm_unreachable("ignore other cases for prototyping");
    });
  }

  void handleTargetDataOpEnteringEdge(omp::TargetDataOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      assert(captureType == omp::VariableCaptureKind::ByRef);
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);  
      bool hasAlloc = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);

      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      assert(!(hasTo && hasAlloc));
      if (hasTo) {
        if (state.refCountRange.second == 0 || hasAlways) {
          // (0, 0)
          state.deviceSubState = state.hostSubState;
        } else {
          if (state.refCountRange.first == 0) {
            // (0, Z+)
            state.deviceSubState = SubState::join(state.deviceSubState, state.hostSubState);
          } else {
            // (Z+, Z+): DO NOTHING
          }
        }
        state.incRefCounts();
        return;
      }
      if (hasAlloc || (hasFrom && !hasTo)) {
        if (state.refCountRange.first == 0) {
          // (0, 0): device -> TOP
          // (0, Z+): device -> join(device, TOP) = TOP
          state.deviceSubState = {SubState::State::Top, nullptr};
        } else {
          // (Z+, Z+): do nothing
        }
        state.incRefCounts();
        return;
      }
      llvm_unreachable("ignore other cases for prototyping");
    });
  }

  void handleTargetDataOpExitingEdge(omp::TargetDataOp op, MemoryAliasLattice *after) {
    opDataUpdate(op.getMapVars(), after, [](MemState& state, omp::ClauseMapFlags direction, omp::VariableCaptureKind captureType){
      assert(captureType == omp::VariableCaptureKind::ByRef);
      bool hasFrom = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::from);
      bool hasTo = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::to);  
      bool hasAlloc = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::storage);

      bool hasAlways = omp::bitEnumContainsAny(direction, omp::ClauseMapFlags::always);
      if (hasFrom) {
        // conditional copy back
        assert(state.refCountRange.first >= 1 && state.refCountRange.second >= 1);
        if (state.refCountRange.second == 1 || hasAlways) {
          // (1, 1) or always: copy back
          state.hostSubState = state.deviceSubState;
        } else {
          if (state.refCountRange.first == 1) {
            // (1, 1+)
            state.hostSubState = SubState::join(state.hostSubState, state.deviceSubState);
          } else {
            // (1+, 1+): do nothing
          }
        }
        state.decRefCounts();
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      if (hasAlloc || (hasTo && !hasFrom)) {
        // DO NOTHING
        state.decRefCounts(); 
        if (state.refCountRange.second == 0) {
          state.deviceSubState.reset();
        }
        return;
      }
      llvm_unreachable("ignore other cases for prototyping");
    });
  }

public:
  TargetFoldabilityAnalysis(DataFlowSolver& solver, func::FuncOp func):
    DenseForwardDataFlowAnalysis(solver), currentFunc(func) {};

  void visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, 
    std::optional<unsigned int> regionFrom, 
    std::optional<unsigned int> regionTo, const 
    MemoryAliasLattice &before, 
    MemoryAliasLattice *after
  ) override {
    auto* op = branch.getOperation();
    assert(op);

    llvm::errs() << "\n[RB] " << op->getName() << " from=";
    if (regionFrom)
      llvm::errs() << *regionFrom;
    else
      llvm::errs() << "parent";

    llvm::errs() << " to=";
    if (regionTo)
      llvm::errs() << *regionTo;
    else
      llvm::errs() << "parent";
    llvm::errs() << "\n";


    MemoryAliasLattice candidate(after->getAnchor());
    candidate.memoryMap = before.memoryMap;

    if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(op)) {
      // Entering:
      // `regionFrom` is nullptr, means it is the parentOP. 
      // `regionTo` is 0, means it is the first region of the Op.
      // In summary, this means from the edge going from the parentOp to the region of the it.
      if (!regionFrom && regionTo && *regionTo == 0) {
        handleTargetOpEnteringEdge(targetOp, &candidate); 
      } else if (regionFrom && *regionFrom == 0 && !regionTo) {
        handleTargetOpExitingEdge(targetOp, &candidate);
      } else {
        llvm_unreachable("Shold only visit the entering edge and the exiting edge.");
      }
      propagateIfChanged(after, after->join(candidate));
      return;
    }
    if (auto targetDataOp = llvm::dyn_cast<omp::TargetDataOp>(op)) {
      if (!regionFrom && regionTo && *regionTo == 0) {
        handleTargetDataOpEnteringEdge(targetDataOp, &candidate);
      } else if (regionFrom && *regionFrom == 0 && !regionTo) {
        handleTargetDataOpExitingEdge(targetDataOp, &candidate);
      } else {
        llvm_unreachable("Shold only visit the entering edge and the exiting edge.");
      }
      propagateIfChanged(after, after->join(candidate));
      return;
    }

    mlir::dataflow::DenseForwardDataFlowAnalysis<MemoryAliasLattice>
      ::visitRegionBranchControlFlowTransfer(branch, regionFrom, regionTo, before, after);
    return;
  }

  // handle standalone operations
  LogicalResult visitOperation(
    mlir::Operation *op,
    const MemoryAliasLattice &before,
    MemoryAliasLattice *after
  ) override { 

    MemoryAliasLattice candidate(after->getAnchor());
    candidate.memoryMap = before.memoryMap;

    if (op->getDialect()->getNamespace() == "arith") {
      // Skip and do not print log
      propagateIfChanged(after, after->join(candidate));
      return mlir::success();
    }

    llvm::TypeSwitch<Operation*>(op)
        .Case([&](omp::TargetUpdateOp targetUpdateData){handleTargetUpdateOp(targetUpdateData, &candidate);})
        .Case([&](omp::TargetEnterDataOp enterOp){handleTargetEnterDataOp(enterOp, &candidate);})
        .Case([&](omp::TargetExitDataOp exitOp){handleTargetExitDataOp(exitOp, &candidate);})
        .Case([&](fir::AllocaOp alloca){
          (&candidate)->memoryMap[alloca.getResult()] = {
            {SubState::State::Bottom, nullptr},
            {SubState::State::Bottom, nullptr},
            {0, 0}
          };
        })
        .Case([&](omp::MapInfoOp info){
          llvm::dbgs() << "\n[DEBUG] Handle: ";
          info.print(llvm::dbgs());
          (&candidate)->print(llvm::dbgs());
          if (info.getMapCaptureType() == omp::VariableCaptureKind::ByCopy) {
            auto varPtrRoot = getRootMem(info.getVarPtr());
            if ((&candidate)->memoryMap.contains(varPtrRoot)) {
              // If byCopy, the root of the result is the result itself.
              (&candidate)->memoryMap[info.getResult()] = (&candidate)->memoryMap.at(varPtrRoot); 
            } else {
              llvm::dbgs() << "\n[DEBUG]Cannot find varPtrRoot in status: ";
              info.print(llvm::dbgs());
              llvm::dbgs() << ", root is: ";
              varPtrRoot.printAsOperand(llvm::dbgs(), {});
              (&candidate)->memoryMap[info.getResult()] = {
                {SubState::State::Top, nullptr},
                {SubState::State::Top, nullptr},
                {0, INT_MAX}
              };
            }
          } else {
            assert(info.getMapCaptureType() == omp::VariableCaptureKind::ByRef);
            // ByRef: mapInfoOp does not change the value of anything, thus do nothing.
          }
        })
        .Case([&](fir::StoreOp storeOp){
          auto rootMem = getRootMem(storeOp.getMemref());
          if ((&candidate)->memoryMap.contains(rootMem)) {
            (&candidate)->memoryMap[rootMem].hostSubState = {SubState::State::Known, storeOp.getValue()};
          } else {
            llvm::dbgs() << "\n[DEBUG]Cannot find rootMem of StoredOp: ";
            storeOp.print(llvm::dbgs());
            llvm::dbgs() << ", rootMem is: ";
            rootMem.printAsOperand(llvm::dbgs(), {});
            (&candidate)->memoryMap[rootMem] = {
              {SubState::State::Top, nullptr},
              {SubState::State::Top, nullptr},
              {0, INT_MAX},
           };
          }
        })
        .Case([&](hlfir::AssignOp assignOp){
          auto rootMem = getRootMem(assignOp.getLhs());
          if ((&candidate)->memoryMap.contains(rootMem)) {
            (&candidate)->memoryMap[rootMem].hostSubState = {SubState::State::Known, assignOp.getRhs()};
          } else {
            llvm::dbgs() << "\n[DEBUG]Cannot find rootMem of assignOp: ";
            assignOp.print(llvm::dbgs());
            llvm::dbgs() << ", rootMem is: ";
            rootMem.printAsOperand(llvm::dbgs(), {});
            (&candidate)->memoryMap[rootMem] = {
              {SubState::State::Top, nullptr},
              {SubState::State::Top, nullptr},
              {0, INT_MAX},
            };
          }
        })
        .Case<omp::TargetOp, omp::TargetDataOp>([&](auto){
          llvm::errs() << "[ERR] Should not appear here: ";
          op->print(llvm::errs(), {});
          llvm::errs() << "\n";
        })
        .Case<omp::MapBoundsOp, func::ReturnOp,
              hlfir::DeclareOp, fir::ConvertOp, fir::LoadOp,
              fir::DummyScopeOp, fir::ShiftOp, fir::AddrOfOp,
              fir::BoxDimsOp, fir::BoxAddrOp, fir::BoxOffsetOp, 
              fir::IsPresentOp>([&](auto){
          // DO NOTHING, prevent from printing too many log records.
        })
        .Default([&](auto){
          llvm::dbgs() << "\nSkipped operation when check foldability:\n";
          op->print(llvm::dbgs(), {});
          llvm::dbgs() << "\n";
        })
    ;
    propagateIfChanged(after, after->join(candidate));
    return success();
  }
  void setToEntryState(MemoryAliasLattice *lattice) override {
    MemoryAliasLattice entryState(lattice->getAnchor());
    // This function is supposed to be only called when initialization. Unless we have external function calls.
    assert(currentFunc && !currentFunc.empty());
    if (!currentFunc || currentFunc.empty()) return;
    for (auto arg: currentFunc.getArguments()) {
      entryState.memoryMap[arg] = {
        {SubState::State::Known, arg},
        {SubState::State::Top, nullptr},
        {0, INT_MAX}
      };
    }
    propagateIfChanged(lattice, lattice->join(entryState));
  }
};

enum struct Foldability: uint8_t {
  Unfoldable, JITFold, AOTConstFold, AOTTempFold
};

static bool isDependOnArgs(Value val, const llvm::DenseSet<Value>& argsSet, mlir::DataFlowSolver& solver) {
  if (argsSet.contains(val)) return true;
  auto* defOp = val.getDefiningOp();
  if (!defOp) {
    llvm::dbgs() << "\n Cannot find definition Op: ";
    val.printAsOperand(llvm::dbgs(), {});
    llvm::dbgs() << "\n";
    std::exit(EXIT_FAILURE);
  }
  if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(defOp)) {
    auto root = getRootMem(loadOp.getMemref());
    auto* lattice = solver.lookupState<MemoryAliasLattice>(solver.getProgramPointBefore(loadOp));
    assert(lattice->memoryMap.contains(root));
    auto state = lattice->memoryMap.at(root);
    assert(state.hostSubState.state == SubState::State::Known);
    return isDependOnArgs(state.hostSubState.value, argsSet, solver);
  } 
  if (defOp->getNumOperands() == 0) {
    if (!llvm::isa<arith::ConstantOp>(defOp)) {
      llvm::errs() << "\nThe val is: \n";
      val.printAsOperand(llvm::errs(), {});
      llvm::errs() << "\n";
      llvm::errs() << "\nThe wild defOp is: ";
      defOp->print(llvm::errs());
      llvm::errs() << "\n";
      std::exit(EXIT_FAILURE);  
    }
    // assert(llvm::isa<arith::ConstantOp>(defOp));
    return false;
  }
  auto acc = false;
  for (const auto& operand : defOp->getOperands()) {
    acc = acc || isDependOnArgs(operand, argsSet, solver);
  };
  return acc;
}

static std::pair<Foldability, Value> checkFoldability(
  Value arg, 
  Value mapVar, 
  omp::TargetOp targetOp, 
  mlir::DataFlowSolver& solver,
  const MemoryAliasLattice* aliasLattice
) {
  // INFO: only fold temporary scalar should already serve our purpose.
  Type eleTy = fir::unwrapRefType(arg.getType());
  if (!eleTy.isIntOrIndexOrFloat()) {
    return {Foldability::Unfoldable, nullptr};
  }

  // INFO: if arg is never declared, we should not fold, as they are metainfo.
  auto declared = false;
  for (Operation* user: arg.getUsers()) {
    if (llvm::isa<hlfir::DeclareOp>(user)) declared = true;
  }
  if (!declared) return {Foldability::Unfoldable, nullptr};

  
  // INFO: if mapVar is arg, we cannot fold.
  auto root = getRootMem(mapVar);
  auto args = targetOp->getParentOfType<func::FuncOp>().getArguments();
  llvm::DenseSet<Value> argsSet(args.begin(), args.end());
  if (argsSet.contains(root)) return {Foldability::Unfoldable, nullptr};

  // INFO: if mapvar is unknown, we can fold in JIT.
  // assert(aliasLattice->memoryMap.contains(root));
  MemState valueState;
  if (aliasLattice->memoryMap.contains(root)) {
    valueState = aliasLattice->memoryMap.at(root);
  } else {
    llvm::dbgs() << "\n[DEBUG] state not find for mapVar: \n";
    mapVar.printAsOperand(llvm::dbgs(), {});
    llvm::dbgs() << ", with root: \n",
    root.printAsOperand(llvm::dbgs(), {});
    valueState = {{SubState::State::Top, nullptr}, {SubState::State::Top, nullptr}, {0, INT_MAX}};
  }

  SubState joinedState;
  if (valueState.refCountRange.first == 0 && valueState.refCountRange.second == 0) {
    // not mapped
    joinedState = valueState.hostSubState;
  } else if (valueState.refCountRange.first == 0) {
    assert(valueState.refCountRange.second > 0);
    joinedState = SubState::join(valueState.hostSubState, valueState.deviceSubState);
  } else {
    assert(valueState.refCountRange.first > 0);
    // already mapped
    joinedState = valueState.deviceSubState;
  }

  if (joinedState.state == SubState::State::Top) return {Foldability::JITFold, joinedState.value}; 
  if (joinedState.state == SubState::State::Bottom) return {Foldability::AOTTempFold, joinedState.value};
  assert(joinedState.state == SubState::State::Known);
  assert(joinedState.value != nullptr);
  llvm::dbgs() << "\nWe're checking :";
  joinedState.value.printAsOperand(llvm::dbgs(), {});
  llvm::dbgs() << "\n";
  if (isDependOnArgs(joinedState.value, argsSet, solver)) {
    return {Foldability::JITFold, joinedState.value};
  }
  return {Foldability::AOTConstFold, joinedState.value};
}

// Print the dataflow chain of val to the target location
static Value materializeValue(
  Value mapVarFinalVal, 
  DataFlowSolver& solver,
  llvm::DenseMap<Value, Value>& cloneCache,
  OpBuilder& opBuilder
) {
  if (cloneCache.contains(mapVarFinalVal)) {
    return cloneCache[mapVarFinalVal];
  }
  auto* defOp = mapVarFinalVal.getDefiningOp();
  if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(defOp)) {
    auto* lattice = solver.lookupState<MemoryAliasLattice>(solver.getProgramPointBefore(loadOp));
    assert(lattice->memoryMap.contains(loadOp.getMemref()));
    auto stat = lattice->memoryMap.at(loadOp.getMemref());
    cloneCache[mapVarFinalVal] = materializeValue(stat.hostSubState.value, solver, cloneCache, opBuilder);
  }

  // If has a corresponding operation.
  IRMapping vMap;
  for (auto operand: defOp->getOperands()) {
    vMap.map(operand, materializeValue(operand, solver, cloneCache, opBuilder));
  }
  auto* newOp = opBuilder.clone(*defOp, vMap);
  cloneCache[mapVarFinalVal] = newOp->getResult(0);
  return cloneCache[mapVarFinalVal];
}
                                                                                                                            
// TODO: not always track back to alloca, maybe track back to address_of
static Value createLocalAlloca(
  Block& entryBlock,
  Value mapVar,
  Value arg,
  OpBuilder& opBuilder,
  omp::TargetOp targetOp
) {
  Type argTy = arg.getType();
  Type eleTy = fir::unwrapRefType(argTy);
  auto newAllocaOp = fir::AllocaOp::create(
    opBuilder, targetOp.getLoc(), eleTy, "", "", ValueRange{}, {}, {});
  Value replacementMem = newAllocaOp.getResult();
  if (replacementMem.getType() != argTy) {
    replacementMem = fir::ConvertOp::create(opBuilder, targetOp.getLoc(), arg.getType(), replacementMem).getResult();
  }
  auto zeroConstant = fir::ZeroOp::create(opBuilder, targetOp.getLoc(), eleTy);
  fir::StoreOp::create(opBuilder, targetOp.getLoc(), zeroConstant.getResult(), replacementMem);
  return replacementMem;
}
                                                                                                                            
static void assignToLocalAlloca(
  Value replacementMem,
  Value mapVarVal,
  omp::TargetOp targetOp,
  DataFlowSolver& solver,
  OpBuilder& opBuilder,
  llvm::DenseMap<Value, Value>& cloneCache
) {
  if (mapVarVal != nullptr) {
    mapVarVal.printAsOperand(llvm::dbgs(), {});
    auto finalVal = materializeValue(mapVarVal, solver, cloneCache, opBuilder);
    fir::StoreOp::create(opBuilder, targetOp.getLoc(), finalVal, replacementMem);
  }
}

static void foldConstantPrivates(
  omp::TargetOp targetOp, 
  OpBuilder& opBuilder,
  mlir::DataFlowSolver& solver
) {
  const auto* aliasLattice = solver.lookupState<MemoryAliasLattice>(solver.getProgramPointBefore(targetOp));
  if (!aliasLattice) return; // INFO: do not fold any private
  Block& entryBlock = targetOp.getRegion().front();
  
  llvm::DenseMap<Value, Value> cloneCache;
  auto mapVarOffset = getBlockMapVarsOffset(targetOp);
  assert(mapVarOffset == 0); // as other argumemts have already been folded inside map_entries
  
  llvm::DenseMap<Value, std::pair<Foldability, Value>> foldabilityMap;
  for (int i = int(targetOp.numMapBlockArgs()) - 1; i >= 0; i--) {
    auto argIdx = mapVarOffset + i; 
    auto arg = entryBlock.getArgument(argIdx);
    auto mapVar = targetOp.getMapVars()[i];
    auto foldabilityRes = checkFoldability(arg, mapVar, targetOp, solver, aliasLattice); 
    foldabilityMap[mapVar] = foldabilityRes;
  }
  // NOTE: DEBUG
  llvm::dbgs() << "\n[DEBUG] Foldability Map: \n";
  for (const auto&[value, pair]: foldabilityMap) {
    llvm::dbgs() << "Value: ";
    value.printAsOperand(llvm::dbgs(), {});
    llvm::dbgs() << ", Foldability: " << std::to_string(int(pair.first)) << " , foldTo: ";
    if (pair.second) {
      pair.second.printAsOperand(llvm::dbgs(), {});
    } else {

    }
    llvm::dbgs() << "\n";
  }
  llvm::dbgs() << "[DEBUG] Foldability Map End \n";

  OpBuilder::InsertionGuard guardAOTFold(opBuilder); 
  opBuilder.setInsertionPointToStart(&entryBlock);
  for (int i = int(targetOp.numMapBlockArgs()) - 1; i >= 0; i--) {
    auto argIdx = mapVarOffset + i; 
    auto arg = entryBlock.getArgument(argIdx);
    auto mapVar = targetOp.getMapVars()[i];
    auto foldabilityRes = foldabilityMap[mapVar];
    auto foldability = foldabilityRes.first;
    if (foldability == Foldability::Unfoldable) {
      continue;
    }
    if (foldability == Foldability::AOTTempFold || foldability == Foldability::AOTConstFold) {
        Value replacementMem = createLocalAlloca(entryBlock, mapVar, arg, opBuilder, targetOp);
      if (foldability == Foldability::AOTConstFold) {
        assignToLocalAlloca(replacementMem, foldabilityRes.second, targetOp, solver, opBuilder, cloneCache);
      }
      arg.replaceAllUsesWith(replacementMem);
      assert(entryBlock.getArgument(argIdx).getUses().empty());
      entryBlock.eraseArgument(argIdx);
      targetOp.getMapVarsMutable().erase(i);
      continue;
    }
  }

  OpBuilder::InsertionGuard guardJITFold(opBuilder); 
  for (int i = int(targetOp.numMapBlockArgs()) - 1; i >= 0; i--) {
    auto argIdx = mapVarOffset + i; 
    auto arg = entryBlock.getArgument(argIdx);
    auto mapVar = targetOp.getMapVars()[i];
    auto foldability = foldabilityMap[mapVar].first;
    if (foldability == Foldability::JITFold) {
      assert(arg.getNumUses() == 1); // should only have a declare operation
      for (auto* user : arg.getUsers()) {
        assert(llvm::isa<hlfir::DeclareOp>(user));
        auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(user);
        opBuilder.setInsertionPointAfter(declareOp);
        auto inType = fir::unwrapRefType(declareOp.getMemref().getType());
        if (declareOp.getShape()) {
          auto shapeOp = llvm::dyn_cast<fir::ShapeOp>(declareOp.getShape().getDefiningOp());
          assert(shapeOp);
          auto allocaOp = fir::AllocaOp::create(
            opBuilder, 
            targetOp.getLoc(), 
            inType,
            /*uniqName=*/ "",
            /*bindcName=*/ "",
            /*typeparams=*/ mlir::ValueRange{},
            {shapeOp.getExtents()}
          );
          auto neoDeclareOp = hlfir::DeclareOp::create(
            opBuilder, 
            targetOp.getLoc(), 
            allocaOp.getResult(), 
            /*uniq_name=*/"", 
            declareOp.getShape(),
            /*typeparams*/ mlir::ValueRange{},
            declareOp.getDummyScope(),
            declareOp.getStorage(),
            declareOp.getStorageOffset(),
            {},
            {},
            {}
          );
          declareOp.getResults().replaceAllUsesWith(neoDeclareOp);
          hlfir::AssignOp::create(
            opBuilder, 
            targetOp.getLoc(), 
            declareOp.getBase(),
            neoDeclareOp.getBase()
          );
        } else {
          // scalar
          auto allocaOp = fir::AllocaOp::create(opBuilder, targetOp.getLoc(), inType, "", "", ValueRange{}, {}, {});
          auto neoDeclareOp = hlfir::DeclareOp::create(opBuilder, targetOp.getLoc(), allocaOp.getResult(), "", {}, {}, declareOp.getDummyScope(), declareOp.getStorage(), declareOp.getStorageOffset(), {}, {}, {});
          declareOp.getResults().replaceAllUsesWith(neoDeclareOp);
          hlfir::AssignOp::create(opBuilder, targetOp.getLoc(), declareOp.getBase(), neoDeclareOp.getBase());
        }
      }
      continue;
    }
  }

  // INFO: we have already declared shadowing alloca above, this is just for making debugging easier.
  llvm::SmallVector<Attribute> jitArgAttrs;
  for (const auto& mapVar: targetOp.getMapVars()) {
    assert(foldabilityMap.contains(mapVar));
    if (foldabilityMap[mapVar].first == Foldability::JITFold) {
      auto strAttr = StringAttr::get(targetOp.getContext());      
      jitArgAttrs.push_back(strAttr);
    } else {
      auto unitAttr = UnitAttr::get(targetOp.getContext());
      jitArgAttrs.push_back(unitAttr);
    }
  }
  targetOp->setAttr("jit_foldability", opBuilder.getArrayAttr(jitArgAttrs));
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
    } else if (auto distributeOp = llvm::dyn_cast<omp::DistributeOp>(owner)) {
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
        llvm::dbgs() << "\nA1:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA1 End\n";
        llvm::SmallVector<int> indicesForAbsorbedHostEvalVars;
        absorbHostEvalVarsToMapEntries(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        removePrivateVarsFromMapEntry(targetOp, opBuilder, indicesForAbsorbedHostEvalVars);
        llvm::dbgs() << "\nA2:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA2 End\n";
        auto wrappers = getOmpWrappers(targetOp);
        flattenTargetOp(wrappers, targetOp, opBuilder);
        llvm::dbgs() << "\nA3:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA3 End\n";

        PassManager pm(&context);
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
        if (failed(pm.run(moduleOp))) {
          llvm::errs() << "Fail to preprocess the moduleOp!\n";
          std::exit(EXIT_FAILURE); // TODO: fall back processsing
        };


        llvm::dbgs() << "\nA3.5:\n"; moduleOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA3.5 End\n";
        DataFlowSolver solver;
        auto funcOp = targetOp->getParentOfType<func::FuncOp>();
        assert(funcOp);
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<dataflow::SparseConstantPropagation>();
        solver.load<TargetFoldabilityAnalysis>(funcOp);
        if (mlir::failed(solver.initializeAndRun(funcOp))) {
          moduleOp.emitError("Dataflow analysis failed to coverage");
          return;
        }

        llvm::dbgs() << "\nA4:\n"; funcOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA4 End\n";
        foldConstantPrivates(targetOp, opBuilder, solver);
        llvm::dbgs() << "\nA5:\n"; funcOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA5 End\n";

        // INFO: Liveness analysis
        PassManager pm2(&context);
        pm2.addPass(mlir::createMem2Reg());
        pm2.addPass(mlir::createCanonicalizerPass());
        if (failed(pm2.run(funcOp))) {
          llvm::errs() << "Fail to preprocess the functionOp!\n";
          std::exit(EXIT_FAILURE); // TODO: fall back processing
        }
        llvm::dbgs() << "\nA6:\n"; funcOp.print(llvm::dbgs()); llvm::dbgs() <<"\nA6 End\n";
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

        //TODO: set attribute to the function argument, this probably not needed as JITFold has already been done!
        if (auto foldabilityArray = targetOp->getAttrOfType<ArrayAttr>("jit_foldability")) {
          assert(f.getNumArguments() == foldabilityArray.size());          
          for (uint i = 0; i < f.getNumArguments(); i++) {
            auto attr = foldabilityArray[i];
            if (llvm::isa<StringAttr>(attr)) {
              f.setArgAttr(i, "jit_xla.foldability", UnitAttr::get(&context));
            }
          }
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
