#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <alloca.h>
#include <cassert>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <variant>

namespace flangomp {
  #define GEN_PASS_DEF_FLATTENOPENMPTARGET
  #include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "flatten-target"

using namespace mlir;

namespace {

static unsigned getPrivateOffset(Operation* wrapper) {
  return 0;
}

static unsigned getHostEvalVarsOffset(Operation* wrapper) {
  if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(wrapper)) {
    return targetOp.numHasDeviceAddrBlockArgs();
  }
  llvm_unreachable("unexpected type for getting hostEvalVars Offset!");
}

static unsigned getMapVarsOffset(Operation* wrapper) {
  if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(wrapper)) {
    return targetOp.numHasDeviceAddrBlockArgs()
      + targetOp.numInReductionBlockArgs()
      + targetOp.numHostEvalBlockArgs();
  }
  llvm_unreachable("unexpected type for getting mapvars Offset!");
}

static unsigned getReductionVarsOffset(Operation* wrapper) {
  return llvm::TypeSwitch<Operation*, unsigned>(wrapper)
    .Case([&](omp::WsloopOp wsLoopOp){
      // [private_vars, reduction_vars]
      return wsLoopOp.numPrivateBlockArgs();
    })
    .Case([&](omp::TeamsOp teamsOp){
      // [private_vars, reduction_vars]
      return teamsOp.numPrivateBlockArgs();
    })
    .Case([&](omp::SimdOp simdOp){
      // [private_vars, reduction_vars]
      return simdOp.numPrivateBlockArgs();
    })
    .Default([&](Operation*) -> unsigned{
      llvm_unreachable("unexpected operation for getting reduction vars offset!");
    });
}

// INFO: Only support private and firstprivate of scalar value.
// template<typename OMPConstruct> 
struct MaterializePrivatePattern: public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  omp::PrivateClauseOp findPrivateRecipe(omp::ParallelOp op, SymbolRefAttr privateSym) const {
    return SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(op, privateSym);
  }

  LogicalResult matchAndRewrite(omp::ParallelOp op, PatternRewriter &rewriter) const {
    auto offset = getPrivateOffset(op);
    if (op.numPrivateBlockArgs() == 0) {
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    for (int i = 0; i < op.numPrivateBlockArgs(); i++) {
      auto privateVar = op.getPrivateVars()[i];
      auto arg = op->getBlock()->getArgument(offset + i); 
      
      auto privateSyms = op.getPrivateSyms();
      assert(privateSyms);
      if (!privateSyms) {
        return failure();
      }
      auto symbol = llvm::dyn_cast<SymbolRefAttr>(privateSyms->getValue()[i]);
      assert(symbol);
      if (!symbol) {
        return failure();
      }
      auto recipe = findPrivateRecipe(op, symbol);
      auto kind = recipe.getDataSharingType();
      Type type = recipe.getType();
      // materializePrivate(kind, type, privateVar, arg);
      // FIXME: should use recipe, kind and type
      rewriter.setInsertionPointToStart(op->getBlock());      
      auto loadOp = fir::LoadOp::create(rewriter, op.getLoc(), privateVar);
      auto local = fir::AllocaOp::create(rewriter, op.getLoc(), type);
      fir::StoreOp::create(rewriter, op.getLoc(), loadOp.getResult(), local.getResult());
      rewriter.replaceAllUsesWith(arg, local.getResult()); 
    }

    // erase
    for (int i = op.numPrivateBlockArgs() - 1; i >= 0; i--) {
      op.getPrivateVarsMutable().erase(i);
      if (auto syms = op.getPrivateSyms()) {
        llvm::SmallVector<mlir::Attribute> newSyms(syms->getValue());
        newSyms.erase(newSyms.begin() + i);
        if (newSyms.empty()) {
          op.removePrivateSymsAttr();
      } else {
          op.setPrivateSymsAttr(rewriter.getArrayAttr(newSyms));
        }
      }
      op.getRegion().front().eraseArgument(offset + i);
    }
    return success();
  };
};

struct MaterializeReductionPattern: public OpRewritePattern<omp::ParallelOp> {
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
};

struct MapHostEvalPattern: public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const {
    auto mapVarsOffset = getMapVarsOffset(op);
    auto hostEvalOffset = getHostEvalVarsOffset(op);
    assert(mapVarsOffset >= hostEvalOffset + op.numHostEvalBlockArgs());
    auto& entryBlock = op.getRegion().front();
    for (unsigned i = 0; i < op.numHostEvalBlockArgs(); i++) {
      auto hostEvalVar = op.getHostEvalVars()[i];
      // MapVars only accept ref args
      // hostEvalVar usually type i32, but we need to assert
      assert(hostEvalVar.getType().isInteger());
      rewriter.setInsertionPoint(op);
      auto preAllocaOp = fir::AllocaOp::create(rewriter, op.getLoc(), hostEvalVar.getType());
      fir::StoreOp::create(rewriter, op.getLoc(), hostEvalVar, preAllocaOp.getResult());
      auto hostEvalVarRef = omp::MapInfoOp::create(
        rewriter, 
        op.getLoc(), 
        preAllocaOp.getType(), 
        preAllocaOp, 
        TypeAttr::get(hostEvalVar.getType()),
        /*ClauseMapFlagsAttr*/omp::ClauseMapFlagsAttr::get(op.getContext(), ::mlir::omp::ClauseMapFlags::implicit),
        /*VariableCaptureKindAttr*/omp::VariableCaptureKindAttr::get(op.getContext(), ::mlir::omp::VariableCaptureKind::ByCopy),
        /*opational: var_ptr_ptr*/{},
        /*ValueRange members*/{},
        /*ArrayAttr members_index*/{},
        /*ValueRange bounds*/{},
        /*FlatSymbolRefAttr mapper_id*/{},
        /*name*/StringAttr::get(op.getContext()),
        /*partial_map=*/BoolAttr::get(op.getContext(), false)
      );

      int insertedArgPosition = mapVarsOffset + op.numMapBlockArgs();
      auto insertedArg = entryBlock.insertArgument(insertedArgPosition, hostEvalVarRef.getResult().getType(), op.getLoc());
      op.getMapVarsMutable().append(hostEvalVarRef.getResult());
      // recover from fir.ref<i32> to i32 and replace original usage of hostEvalVars
      rewriter.setInsertionPointToStart(&entryBlock);
      auto insertedArgVal = fir::LoadOp::create(rewriter, op.getLoc(), insertedArg);
      entryBlock.getArgument(i + hostEvalOffset).replaceAllUsesWith(insertedArgVal);
    }

    for (int i = op.numHostEvalBlockArgs() - 1; i >= 0; i--) {
      op.getHostEvalVarsMutable().erase(i);
      entryBlock.eraseArgument(hostEvalOffset + i);
    }
  }
};

struct ReplaceLoopPattern: public OpRewritePattern<omp::LoopNestOp> {
  using OpRewritePattern<omp::LoopNestOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::LoopNestOp op, PatternRewriter &rewriter) const {
    fir::DoLoopOp outmostLoopOp = nullptr;
    fir::DoLoopOp lastLoopOp = nullptr;
    llvm::SmallVector<mlir::Value> newInductionVars;
    // from outmost to innermost
    for (uint64_t i = 0; i < op.getCollapseNumLoops(); i ++) {
      if (!outmostLoopOp) {
        rewriter.setInsertionPoint(op);
      } else {
        rewriter.setInsertionPointToStart(lastLoopOp.getBody());
      }
      auto convertToIndexIfNot = [&](Value val) -> Value {
        if (!val.getType().isIndex()) {
          auto convertOp = fir::ConvertOp::create(rewriter, op.getLoc(), IndexType::get(rewriter.getContext()), val, {});
          return convertOp.getResult();
        }
        return val;
      };
      auto lb = convertToIndexIfNot(op.getLoopLowerBounds()[i]);
      auto ub = convertToIndexIfNot(op.getLoopUpperBounds()[i]);
      auto step = convertToIndexIfNot(op.getLoopSteps()[i]);

      auto loopOp = fir::DoLoopOp::create(rewriter, op.getLoc(), lb, ub, step);
      if (outmostLoopOp == nullptr) {
        outmostLoopOp = loopOp;
      }
      lastLoopOp = loopOp;
      newInductionVars.push_back(loopOp.getInductionVar()); 
    }

    assert(lastLoopOp);

    auto* targetBlock = lastLoopOp.getBody();
    auto* sourceBlock = &op.getRegion().front();

    for (uint64_t i = 0; i < op.getCollapseNumLoops(); i++) {
      auto loopIV = newInductionVars[i];
      auto lNOpIV = op.getRegion().front().getArgument(i);
      if (loopIV.getType() != lNOpIV.getType()) {
        rewriter.setInsertionPointToStart(lastLoopOp.getBody());
        auto loopIVConvertOp = fir::ConvertOp::create(rewriter, op.getLoc(), lNOpIV.getType(), loopIV, {});
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
    rewriter.eraseOp(op);
    assert(outmostLoopOp);
    return success();
  }
};

struct NestedStructurePattern: public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;

  // FIXME: should cover all omp operations with regions.
  /// Get wrappers ordering from outmost to innermost
  SmallVector<Operation*> getOmpWrappers(omp::TargetOp targetOp) const {
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

  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const {
    auto wrappers = getOmpWrappers(op);
    if (wrappers.empty()) return failure();
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
    return success();
  }
};


class FlattenTargetPass : public flangomp::impl::FlattenOpenMPTargetBase<FlattenTargetPass> {
public:
  void runOnOperation() override {
    return;   
  };
};
}// namespace



