#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
#include <mlir/IR/Builders.h>
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

  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const {
    unsigned mapVarArgOffset = getMapVarsOffset(op); 
    auto& entryBlock = op.getRegion().getBlocks().front(); 

    llvm::SmallVector<int> indicesToMaterialize;
    for (int i = 0; i < op.numMapBlockArgs(); i++) {
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
      assert(mapVar.getType().isIntOrIndexOrFloat()); 
    }

    if (indicesToMaterialize.size() == 0) {
      return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    for (const int idx: indicesToMaterialize) {
      rewriter.setInsertionPointToStart(&entryBlock);
      auto mapVar = op.getMapVars()[idx]; 
      auto mapVarArg = entryBlock.getArgument(idx + mapVarArgOffset);
      auto allocaOp = fir::AllocaOp::create(rewriter, op.getLoc(), mapVarArg.getType());
      auto loadOp = fir::LoadOp::create(rewriter, op.getLoc(), mapVar);
      fir::StoreOp::create(rewriter, op.getLoc(), loadOp.getResult(), allocaOp.getResult());
      rewriter.replaceAllUsesWith(mapVarArg, allocaOp.getResult());
      assert(mapVarArg.getUses().empty());
    }

    return failure();
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


