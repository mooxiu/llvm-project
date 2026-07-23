// Regressional Test: flang/test/Transforms/OpenMP/materialize-implicit-private.mlir

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace flangomp {
  #define GEN_PASS_DEF_MATERIALIZEPRIVATES
  #include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "materialize-privates"

using namespace mlir;

namespace {
static unsigned getMapVarsOffset(Operation* wrapper) {
  if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(wrapper)) {
    return targetOp.numHasDeviceAddrBlockArgs()
      + targetOp.numInReductionBlockArgs()
      + targetOp.numHostEvalBlockArgs();
  }
  llvm_unreachable("unexpected type for getting mapvars Offset!");
}

struct MaterializeImplicitPrivatesPattern: public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;

  // TODO: should cover all the situations that I support
  bool isDeclaredLocally(Value val) const {
    auto* defOp = val.getDefiningOp();
    if (!defOp) {
      return false;
    }
    if (llvm::isa<fir::AllocaOp>(defOp)) {
      return true;
    }
    if (auto declareOp = llvm::dyn_cast<hlfir::DeclareOp>(defOp)) {
      return isDeclaredLocally(declareOp.getMemref());
    }
    if (auto mapInfoOp = llvm::dyn_cast<omp::MapInfoOp>(defOp)) {
      return isDeclaredLocally(mapInfoOp.getVarPtr());
    }
    return false;
  }

  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const override {
    unsigned mapVarArgOffset = getMapVarsOffset(op); 
    auto& entryBlock = op.getRegion().getBlocks().front(); 

    llvm::SmallVector<int> indicesToMaterialize;
    for (unsigned i = 0; i < op.numMapBlockArgs(); i++) {
      auto mapVar = op.getMapVars()[i]; 
      auto mapVarArg = entryBlock.getArgument(i + mapVarArgOffset);
      
      // a mapVar must be defined by an MapInfoOp (verification logic)
      auto mapInfo = llvm::dyn_cast<omp::MapInfoOp>(mapVar.getDefiningOp());
      assert(mapInfo);
      if (mapInfo.getMapCaptureType() != omp::VariableCaptureKind::ByCopy) {
        continue;
      }
      bool hasDeclareOp = false;
      for (Operation* user : mapVarArg.getUsers()) {
        if (llvm::isa<hlfir::DeclareOp>(user)) {
          hasDeclareOp = true;
          break;
        }
      }
      if (!hasDeclareOp) continue;

      if (!isDeclaredLocally(mapVar)) {
        continue;
      }

      assert(llvm::isa<fir::ReferenceType>(mapVarArg.getType()));
      auto argEleType = fir::unwrapRefType(mapVarArg.getType());
      assert(argEleType.isIntOrIndexOrFloat());

      indicesToMaterialize.push_back(i);
    }

    if (indicesToMaterialize.size() == 0) {
      return failure();
    }

    llvm::errs() << "\n[DEBUG]We have " << indicesToMaterialize.size() << " indices to materialize.";
    for (const int idx: indicesToMaterialize) {
      llvm::errs() << "\n[DEBUG]Idx to materialize: " << idx;
      rewriter.setInsertionPointToStart(&entryBlock);
      // auto mapVar = op.getMapVars()[idx]; 
      auto mapVarArg = entryBlock.getArgument(idx + mapVarArgOffset);
      llvm::errs() << "\n[DEBUG]MapVarArg:";
      mapVarArg.printAsOperand(llvm::errs(), {});
      llvm::errs() << "\n";
      
      assert(llvm::isa<fir::ReferenceType>(mapVarArg.getType()));
      auto argEleType = fir::unwrapRefType(mapVarArg.getType());

      auto allocaOp = fir::AllocaOp::create(rewriter, op.getLoc(), argEleType);
      auto loadOp = fir::LoadOp::create(rewriter, op.getLoc(), mapVarArg);
      fir::StoreOp::create(rewriter, op.getLoc(), loadOp.getResult(), allocaOp.getResult());
      rewriter.replaceAllUsesExcept(mapVarArg, allocaOp.getResult(), loadOp);
    }
    return success();
  }
};

class MaterializePrivatesPass 
  : public flangomp::impl::MaterializePrivatesBase<MaterializePrivatesPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp->getContext();
    OpBuilder opBuilder(ctx);

    RewritePatternSet patterns(ctx);
    patterns.add<MaterializeImplicitPrivatesPattern>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();

    FrozenRewritePatternSet frozenRewritePatternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(moduleOp, frozenRewritePatternSet, config))) {
      signalPassFailure();
      return;
    }
    return; 
  }
};
} // namespace


