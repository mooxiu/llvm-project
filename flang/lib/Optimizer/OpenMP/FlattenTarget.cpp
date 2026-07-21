// Regressional Test: 

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <alloca.h>
#include <cassert>
#include <mlir/Dialect/OpenMP/OpenMPClauseOperands.h>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPInterfaces.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <string>
#include <utility>

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
template<typename T> 
struct MaterializePrivatePattern: public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  omp::PrivateClauseOp findPrivateRecipe(T op, SymbolRefAttr privateSym) const {
    return SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(op, privateSym);
  }

  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const {
    auto offset = getPrivateOffset(op);
    if (op.numPrivateBlockArgs() == 0) {
      return failure();
    }

    Block& entryBlock = op.getRegion().getBlocks().front();
    OpBuilder::InsertionGuard guard(rewriter);
    for (unsigned i = 0; i < op.numPrivateBlockArgs(); i++) {
      auto privateVar = op.getPrivateVars()[i];
      auto arg = entryBlock.getArgument(offset + i); 
      
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

struct MaterializeReductionPattern: public OpRewritePattern<omp::LoopOp> {
  using OpRewritePattern<omp::LoopOp>::OpRewritePattern;

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


  omp::ReductionClauseOps getReductionRecipe() {
    llvm_unreachable("IMPLEMENT ME!");
  }

  LogicalResult matchAndRewrite(omp::LoopOp op, PatternRewriter &rewriter) const override {
    if (op.numReductionBlockArgs() == 0) {
      return failure();
    }

    auto reductionVarOffset = getReductionVarsOffset(op);
    auto& entryBlock = op.getRegion().getBlocks().front();
    for (int i = 0; i < op.numReductionBlockArgs(); i++) {
      auto reductionVar = op.getReductionVars()[i];
      auto arg =  entryBlock.getArgument(reductionVarOffset + i);
      
      auto allocaOp = fir::AllocaOp::create(rewriter, op.getLoc(), arg.getType());
      auto loadOp = fir::LoadOp::create(rewriter, op.getLoc(), reductionVar);
      fir::StoreOp::create(rewriter, op.getLoc(), loadOp.getResult(), allocaOp.getResult());

      rewriter.replaceAllUsesWith(arg, allocaOp.getResult());
      // TODO: should we store the value back? 
      // TODO: how should we do with the operation?
      // FIXME: Implement me!
    }

  };
};

struct MapHostEvalPattern: public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const {
    if (op.numHostEvalBlockArgs() == 0) {
      return failure();
    };
    auto mapVarsOffset = getMapVarsOffset(op);
    auto hostEvalOffset = getHostEvalVarsOffset(op);
    assert(mapVarsOffset >= hostEvalOffset + op.numHostEvalBlockArgs());
    auto& entryBlock = op.getRegion().front();
    for (unsigned i = 0; i < op.numHostEvalBlockArgs(); i++) {
      auto hostEvalVar = op.getHostEvalVars()[i];
      // MapVars only accept ref args
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
    return success();
  }
};

struct ReplaceLoopPattern: public OpRewritePattern<omp::WsloopOp> {
  using OpRewritePattern<omp::WsloopOp>::OpRewritePattern;

  static bool isTrivialWsloop(omp::WsloopOp op) {
    // Operand-based clauses, such as private/reduction/linear.
    if (op->getNumOperands() != 0)
      return false;
    return true;
  }

  static Value convertToIndexIfNot(Value val, Location loc, PatternRewriter &rewriter) {
    if (!val.getType().isIndex()) {
      auto convertOp = fir::ConvertOp::create(rewriter, loc, IndexType::get(rewriter.getContext()), val, {});
      return convertOp.getResult();
    }
    return val;
  };

  //FIX: should actually translate into something like scf::ParallelOp as wsloop means a prallel execution rather than serial loops.
  LogicalResult replaceNestLoopOp(
    omp::LoopNestOp op, 
    omp::WsloopOp wsOp,
    PatternRewriter &rewriter
  ) const {
    if (!op.getLoopInclusive()) return rewriter.notifyMatchFailure(op, "non-inclusive loop unsupported");
    if (!op.getRegion().hasOneBlock()) return rewriter.notifyMatchFailure(op, "multi-block loop unsupported");

    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = op.getLoc();
    fir::DoLoopOp outmostLoopOp = nullptr;
    fir::DoLoopOp lastLoopOp = nullptr;
    llvm::SmallVector<mlir::Value> newInductionVars;
    // from outmost to innermost
    for (uint64_t i = 0; i < op.getCollapseNumLoops(); i ++) {
      rewriter.setInsertionPoint(wsOp);
      auto lb = convertToIndexIfNot(op.getLoopLowerBounds()[i], loc, rewriter);
      auto ub = convertToIndexIfNot(op.getLoopUpperBounds()[i], loc, rewriter);
      auto step = convertToIndexIfNot(op.getLoopSteps()[i], loc, rewriter);

      if (!outmostLoopOp) {
        rewriter.setInsertionPoint(op);
      } else {
        rewriter.setInsertionPointToStart(lastLoopOp.getBody());
      }
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

    rewriter.eraseOp(sourceBlock->getTerminator());

    rewriter.inlineBlockBefore(
        sourceBlock,
        targetBlock,
        targetBlock->getTerminator()->getIterator(),
        newInductionVars);

    rewriter.eraseOp(op);
    assert(outmostLoopOp);
    return success();
  }

  LogicalResult matchAndRewrite(omp::WsloopOp op, PatternRewriter &rewriter) const {
    if (llvm::isa<omp::DistributeOp, omp::TeamsOp>(op->getParentOp())) {
      return failure();
    }
    assert(op.getRegion().hasOneBlock());
    auto& block = op.getRegion().getBlocks().front();
    assert(block.getOperations().size() == 1);
    Operation& firstOp = block.getOperations().front();
    llvm::DenseSet<StringRef> transformedNamespaces = {"scf", "fir"};

    if (auto nestLoopOp = llvm::dyn_cast<omp::LoopNestOp>(&firstOp)) {
      auto res = replaceNestLoopOp(nestLoopOp, op, rewriter);
      if (res.failed()) {
        return failure();
      }
    } else if (transformedNamespaces.contains(firstOp.getDialect()->getNamespace())) {
      return failure();
    } else {
      llvm::errs() << "\nUnexpected Operation: ";
      firstOp.dump();
      return failure();
    }

    if (isTrivialWsloop(op)) {
      rewriter.inlineBlockBefore(
        &block,
        op->getBlock(),
        op->getIterator()
      );
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

template<typename T>
struct NestedStructurePattern: public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const {
    if (!op) {
      return failure();
    }
    if (op.getNumOperands() > 0) {
      return failure();
    }
    if (!op.getRegion().hasOneBlock()) {
      return failure();
    }
    if (op->getDialect()->getNamespace()!="omp") {
      return failure();
    }

    Operation* parentOp = op->getParentOp();
    auto* parentBlock = op->getBlock();
    auto& innerBlock = op.getRegion().getBlocks().front(); 
    if (!innerBlock.empty() && innerBlock.back().template hasTrait<mlir::OpTrait::IsTerminator>()) {
      rewriter.eraseOp(&innerBlock.back());
    }
    rewriter.inlineBlockBefore(
      &innerBlock, 
      parentBlock,
      op->getIterator()
    );
    rewriter.eraseOp(op);

    // Delete omp.composite if necessary
    if (!parentOp->getDiscardableAttr("omp.composite")) {
      return success();
    }
    auto stillContainsLoopWrapper = [&](Operation* op) -> bool {
      bool contains = false;; 
      op->walk([&](Operation* subOp){
        if (subOp->getDialect()->getNamespace() != "omp") {
          return WalkResult::skip();
        }
        if (isa<omp::LoopWrapperInterface>(op->getParentOp())) {
          contains = true;
          return WalkResult::interrupt();
        }
        return WalkResult::skip();
      });
      return contains;
    }; 
    if (!stillContainsLoopWrapper(parentOp)) {
      rewriter.modifyOpInPlace(parentOp, [&]{
        parentOp->removeDiscardableAttr("omp.composite");
      });
    }
    return success();
  }
};

class FlattenTargetPass : public flangomp::impl::FlattenOpenMPTargetBase<FlattenTargetPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp->getContext();
    OpBuilder opBuilder(ctx);

    RewritePatternSet patterns(ctx);
    patterns.add<MaterializePrivatePattern<omp::ParallelOp>>(ctx);
    patterns.add<MapHostEvalPattern>(ctx);
    patterns.add<ReplaceLoopPattern>(ctx);
    patterns.add<NestedStructurePattern<omp::ParallelOp>>(ctx);
    patterns.add<NestedStructurePattern<omp::DistributeOp>>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();

    FrozenRewritePatternSet frozenRewritePatternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(moduleOp, frozenRewritePatternSet, config))) {
      signalPassFailure();
      return;
    }
    //
    // RewritePatternSet deNestPatterns(ctx);
    // deNestPatterns.add<NestedStructurePattern>(ctx);
    // FrozenRewritePatternSet deNestPatternSet(std::move(deNestPatterns));
    // if (failed(applyPatternsGreedily(moduleOp, deNestPatternSet, config))) {
    //   signalPassFailure();
    //   return;
    // }

    return;
  };
};

}// namespace



