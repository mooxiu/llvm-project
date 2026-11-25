// Regressional Test: flang/test/Transforms/OpenMP/flatten-target.mlir

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <alloca.h>
#include <cassert>
#include <mlir/Dialect/Arith/IR/Arith.h>
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
#include <optional>
#include <utility>

namespace flangomp {
#define GEN_PASS_DEF_FLATTENOPENMPTARGET
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "flatten-target"

using namespace mlir;

static unsigned getPrivateOffset(Operation *wrapper) {
  // FIX: IMPLEMENT ME
  return 0;
}

static unsigned getHostEvalVarsOffset(Operation *wrapper) {
  if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(wrapper)) {
    return targetOp.numHasDeviceAddrBlockArgs();
  }
  llvm_unreachable("unexpected type for getting hostEvalVars Offset!");
}

static unsigned getMapVarsOffset(Operation *wrapper) {
  if (auto targetOp = llvm::dyn_cast<omp::TargetOp>(wrapper)) {
    return targetOp.numHasDeviceAddrBlockArgs() +
           targetOp.numInReductionBlockArgs() + targetOp.numHostEvalBlockArgs();
  }
  llvm_unreachable("unexpected type for getting mapvars Offset!");
}

static unsigned getReductionVarsOffset(Operation *wrapper) {
  return llvm::TypeSwitch<Operation *, unsigned>(wrapper)
      .Case([&](omp::WsloopOp wsLoopOp) {
        // [private_vars, reduction_vars]
        return wsLoopOp.numPrivateBlockArgs();
      })
      .Case([&](omp::TeamsOp teamsOp) {
        // [private_vars, reduction_vars]
        return teamsOp.numPrivateBlockArgs();
      })
      .Case([&](omp::SimdOp simdOp) {
        // [private_vars, reduction_vars]
        return simdOp.numPrivateBlockArgs();
      })
      .Default([&](Operation *) -> unsigned {
        llvm_unreachable(
            "unexpected operation for getting reduction vars offset!");
      });
}

static mlir::Type unwrapElementType(mlir::Type type) {
  while (true) {
    if (auto refTy = llvm::dyn_cast<fir::ReferenceType>(type)) {
      type = refTy.getEleTy();
      continue;
    }
    if (auto ptrTy = llvm::dyn_cast<fir::PointerType>(type)) {
      type = ptrTy.getEleTy();
      continue;
    }
    if (auto heapTy = llvm::dyn_cast<fir::HeapType>(type)) {
      type = heapTy.getEleTy();
      continue;
    }
    if (auto boxTy = llvm::dyn_cast<fir::BaseBoxType>(type)) {
      type = boxTy.getEleTy();
      continue;
    }
    if (auto seqTy = llvm::dyn_cast<fir::SequenceType>(type)) {
      type = seqTy.getEleTy();
      continue;
    }
    if (auto shapedTy = llvm::dyn_cast<mlir::ShapedType>(type)) {
      type = shapedTy.getElementType();
      continue;
    }
    return type;
  }
}

template <typename T>
static omp::DeclareReductionOp findReductionRecipe(T op, unsigned index) {
  std::optional<ArrayAttr> syms = op.getReductionSyms();
  assert(syms);
  assert(index < syms->size());
  auto symbol = cast<SymbolRefAttr>((*syms)[index]);
  return SymbolTable::lookupNearestSymbolFrom<omp::DeclareReductionOp>(
      op.getOperation(), symbol);
}

static Value getInitializedValue(omp::DeclareReductionOp recipe,
                                 PatternRewriter &rewriter) {
  auto &initRegion = recipe.getInitializerRegion();
  Block &initBlock = initRegion.front();

  auto initYield = dyn_cast<omp::YieldOp>(initBlock.getTerminator());

  if (!initYield || initYield->getNumOperands() != 1 ||
      !llvm::hasSingleElement(initBlock.without_terminator())) {
    return nullptr;
  }

  Value initializedValue = initYield.getOperand(0);
  auto initializer = initializedValue.getDefiningOp<arith::ConstantOp>();
  assert(initializer);
  Operation *cloned = rewriter.clone(*initializer.getOperation());
  return cloned->getResult(0);
  return initializedValue;
}

static std::optional<arith::AtomicRMWKind> classifyCombiner(Operation *op) {
  if (llvm::isa<arith::AddIOp>(op))
    return arith::AtomicRMWKind::addi;
  if (llvm::isa<arith::AddFOp>(op))
    return arith::AtomicRMWKind::addf;
  if (llvm::isa<arith::MulIOp>(op))
    return arith::AtomicRMWKind::muli;
  if (llvm::isa<arith::MulFOp>(op))
    return arith::AtomicRMWKind::mulf;
  if (llvm::isa<arith::MaxNumFOp>(op))
    return arith::AtomicRMWKind::maxnumf;
  if (llvm::isa<arith::MinNumFOp>(op))
    return arith::AtomicRMWKind::minnumf;
  llvm::errs() << "\nUnexpected Reduction Combiner: ";
  op->dump();
  return std::nullopt;
}

// TODO: in fact it reduce can accept complicated reductions, but we're only
// going to accept Add, Mul, MAXF, MINF Can be expanded but also need to expand
// Enzyme-JAX.
static std::optional<arith::AtomicRMWKind>
getCombiner(omp::DeclareReductionOp recipe) {
  auto &reductionRegion = recipe.getReductionRegion();
  Block &reductionBlock = reductionRegion.front();

  auto reductionYield = dyn_cast<omp::YieldOp>(reductionBlock.getTerminator());

  if (!reductionYield || reductionYield->getNumOperands() != 1 ||
      reductionBlock.getNumArguments() != 2 ||
      !llvm::hasSingleElement(reductionBlock.without_terminator())) {
    return std::nullopt;
  }

  Operation *combiner = reductionYield.getOperand(0).getDefiningOp();

  if (!combiner || combiner->getNumOperands() != 2 ||
      combiner->getNumResults() != 1 ||
      combiner->getOperand(0) != reductionBlock.getArgument(0) ||
      combiner->getOperand(1) != reductionBlock.getArgument(1))
    return std::nullopt;

  return classifyCombiner(combiner);
}

static Value combineReduction(arith::AtomicRMWKind kind, Value localAcc,
                              Value globalAcc, Location loc,
                              PatternRewriter &rewriter) {
  if (kind == arith::AtomicRMWKind::addi) {
    return arith::AddIOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  if (kind == arith::AtomicRMWKind::addf) {
    return arith::AddFOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  if (kind == arith::AtomicRMWKind::muli) {
    return arith::MulIOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  if (kind == arith::AtomicRMWKind::mulf) {
    return arith::MulFOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  if (kind == arith::AtomicRMWKind::maxnumf) {
    return arith::MaxNumFOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  if (kind == arith::AtomicRMWKind::minnumf) {
    return arith::MinNumFOp::create(rewriter, loc, localAcc, globalAcc)
        .getResult();
  }
  llvm_unreachable("unexpected kind");
}

struct MapHostEvalPattern : public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::TargetOp op,
                                PatternRewriter &rewriter) const override {
    if (op.numHostEvalBlockArgs() == 0) {
      return failure();
    };
    auto mapVarsOffset = getMapVarsOffset(op);
    auto hostEvalOffset = getHostEvalVarsOffset(op);
    assert(mapVarsOffset >= hostEvalOffset + op.numHostEvalBlockArgs());
    auto &entryBlock = op.getRegion().front();
    for (unsigned i = 0; i < op.numHostEvalBlockArgs(); i++) {
      auto hostEvalVar = op.getHostEvalVars()[i];
      // MapVars only accept ref args
      assert(hostEvalVar.getType().isInteger());
      rewriter.setInsertionPoint(op);
      auto preAllocaOp =
          fir::AllocaOp::create(rewriter, op.getLoc(), hostEvalVar.getType());
      fir::StoreOp::create(rewriter, op.getLoc(), hostEvalVar,
                           preAllocaOp.getResult());
      auto hostEvalVarRef = omp::MapInfoOp::create(
          rewriter, op.getLoc(), preAllocaOp.getType(), preAllocaOp,
          TypeAttr::get(hostEvalVar.getType()),
          /*ClauseMapFlagsAttr*/
          omp::ClauseMapFlagsAttr::get(op.getContext(),
                                       ::mlir::omp::ClauseMapFlags::implicit),
          /*VariableCaptureKindAttr*/
          omp::VariableCaptureKindAttr::get(
              op.getContext(), ::mlir::omp::VariableCaptureKind::ByCopy),
          /*opational: var_ptr_ptr*/ {},
          /*ValueRange members*/ {},
          /*ArrayAttr members_index*/ {},
          /*ValueRange bounds*/ {},
          /*FlatSymbolRefAttr mapper_id*/ {},
          /*name*/ StringAttr::get(op.getContext()),
          /*partial_map=*/BoolAttr::get(op.getContext(), false));

      int insertedArgPosition = mapVarsOffset + op.numMapBlockArgs();
      auto insertedArg = entryBlock.insertArgument(
          insertedArgPosition, hostEvalVarRef.getResult().getType(),
          op.getLoc());
      op.getMapVarsMutable().append(hostEvalVarRef.getResult());
      // recover from fir.ref<i32> to i32 and replace original usage of
      // hostEvalVars
      rewriter.setInsertionPointToStart(&entryBlock);
      auto insertedArgVal =
          fir::LoadOp::create(rewriter, op.getLoc(), insertedArg);
      entryBlock.getArgument(i + hostEvalOffset)
          .replaceAllUsesWith(insertedArgVal);
    }

    for (int i = op.numHostEvalBlockArgs() - 1; i >= 0; i--) {
      op.getHostEvalVarsMutable().erase(i);
      entryBlock.eraseArgument(hostEvalOffset + i);
    }
    return success();
  }
};

template <typename T>
static void materializePrivatesImpl(T op, PatternRewriter &rewriter) {
  auto iface = dyn_cast<omp::BlockArgOpenMPOpInterface>(op.getOperation());
  if (!iface)
    return;

  auto offset = getPrivateOffset(op);
  if (iface.numPrivateBlockArgs() == 0) {
    return;
  }

  auto privateVarNum = iface.numPrivateBlockArgs();
  auto loc = iface->getLoc();

  Block &entryBlock = op->getRegion(0).getBlocks().front();
  for (unsigned i = 0; i < privateVarNum; i++) {
    auto privateVar = iface.getPrivateVars()[i];
    auto arg = entryBlock.getArgument(offset + i);

    auto privateSyms = op.getPrivateSyms();
    assert(privateSyms);
    auto symbol = llvm::dyn_cast<SymbolRefAttr>(privateSyms->getValue()[i]);
    assert(symbol);

    auto recipe = SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
        iface, symbol);
    auto kind = recipe.getDataSharingType();
    Type type = recipe.getType();
    // materializePrivate(kind, type, privateVar, arg);
    // FIXME: should use recipe, kind and type, but only consider scalar for now
    rewriter.setInsertionPointToStart(&entryBlock);
    auto loadOp = fir::LoadOp::create(rewriter, loc, privateVar);
    auto local = fir::AllocaOp::create(rewriter, loc, type);
    fir::StoreOp::create(rewriter, loc, loadOp.getResult(), local.getResult());
    rewriter.replaceAllUsesWith(arg, local.getResult());
  }

  // erase
  for (int i = privateVarNum - 1; i >= 0; i--) {
    rewriter.modifyOpInPlace(op, [&] {
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
    });
  }
  return;
}

static void materializePrivates(Operation *op, PatternRewriter &rewriter) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<omp::ParallelOp, omp::DistributeOp, omp::TeamsOp, omp::WsloopOp,
            omp::LoopOp, omp::SimdOp>([&](auto typedOp) {
        return materializePrivatesImpl(typedOp, rewriter);
      })
      .Default([&](Operation *) {});
}

template <typename T>
void materializeReductionsImpl(T op, PatternRewriter &rewriter) {
  auto iface = cast<omp::BlockArgOpenMPOpInterface>(op.getOperation());
  unsigned reductionVarNum = iface.numReductionBlockArgs();
  if (reductionVarNum == 0) {
    return;
  }

  Block &entryBlock = op->getRegion(0).getBlocks().front();
  Location loc = op->getLoc();
  auto reductionOffset = getReductionVarsOffset(op);
  for (unsigned i = 0; i < reductionVarNum; i++) {
    omp::DeclareReductionOp recipe = findReductionRecipe(op, i);
    assert(recipe);

    auto reduceVar = iface.getReductionVars()[i];
    auto reduceArg = entryBlock.getArgument(i + reductionOffset);

    rewriter.setInsertionPointToStart(&entryBlock);
    auto reduceAccType = unwrapElementType(reduceArg.getType());
    assert(reduceAccType.isIntOrFloat());
    auto alloca = fir::AllocaOp::create(rewriter, loc, reduceAccType);
    auto initVal = getInitializedValue(recipe, rewriter);
    fir::StoreOp::create(rewriter, loc, initVal, alloca);
    rewriter.replaceAllUsesWith(reduceArg, alloca.getResult());

    if (entryBlock.mightHaveTerminator()) {
      if (entryBlock.getTerminator()) {
        rewriter.setInsertionPoint(entryBlock.getTerminator());
      } else {
        rewriter.setInsertionPointToEnd(&entryBlock);
      }
    } else {
      rewriter.setInsertionPointToEnd(&entryBlock);
    }
    auto localAcc =
        fir::LoadOp::create(rewriter, loc, alloca.getResult()).getResult();
    auto globalAcc = fir::LoadOp::create(rewriter, loc, reduceVar).getResult();
    assert(getCombiner(recipe).has_value());
    auto kind = getCombiner(recipe).value();
    auto res = combineReduction(kind, localAcc, globalAcc, loc, rewriter);
    fir::StoreOp::create(rewriter, loc, res, reduceVar);
  }

  rewriter.modifyOpInPlace(op, [&] {
    op.getReductionVarsMutable().clear();
    op->removeAttr("reduction_syms");
    op->removeAttr("reduction_byref");
    op->removeAttr("reduction_mod");

    for (int i = reductionVarNum - 1; i >= 0; i--)
      entryBlock.eraseArgument(reductionOffset + i);
  });

  return;
}

static void materializeReductions(Operation *op, PatternRewriter &rewriter) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<omp::ParallelOp, omp::TeamsOp, omp::WsloopOp, omp::LoopOp,
            omp::SimdOp>([&](auto typedOp) {
        return materializeReductionsImpl(typedOp, rewriter);
      })
      .Default([&](Operation *) {});
}

static void promoteAndDelete(Operation *op, PatternRewriter &rewriter) {
  Block &body = op->getRegion(0).front();
  if (body.mightHaveTerminator()) {
    if (Operation *terminator = body.getTerminator()) {
      rewriter.eraseOp(terminator);
    }
  }
  rewriter.inlineBlockBefore(&body, op, ValueRange{});
  rewriter.eraseOp(op);
  return;
}

namespace {
// We're going to deal with:
// omp::TargetOp,
// omp::TeamsOp,
// omp::DistributeOp,
// omp::ParallelOp,
// omp::WsloopOp,
// omp::LoopNestOp,
// omp::LoopOp
// omp::SimdOp

struct FlattenPattern : public OpRewritePattern<omp::LoopNestOp> {
  using OpRewritePattern<omp::LoopNestOp>::OpRewritePattern;

  // Wrappers does not contain LoopNestOp and TargetOp
  static llvm::SmallVector<Operation *> collectWrappers(omp::LoopNestOp op) {
    llvm::SmallVector<Operation *> wrappers;
    Operation *curr = op;
    while (true) {
      Operation *parentOp = curr->getParentOp();
      // there can be some store operation in between
      if (llvm::isa<omp::TargetOp>(parentOp) ||
          parentOp->getDialect()->getNamespace() != "omp") {
        break;
      }
      wrappers.push_back(parentOp);
      curr = parentOp;
    }
    return wrappers;
  }

  static Value convertToIndexIfNot(Value val, Location loc,
                                   PatternRewriter &rewriter) {
    if (!val.getType().isIndex()) {
      auto convertOp = fir::ConvertOp::create(
          rewriter, loc, IndexType::get(rewriter.getContext()), val, {});
      return convertOp.getResult();
    }
    return val;
  };

  LogicalResult replaceNestLoopOp(omp::LoopNestOp op,
                                  PatternRewriter &rewriter) const {

    if (!op.getLoopInclusive())
      return rewriter.notifyMatchFailure(op, "non-inclusive loop unsupported");
    if (!op.getRegion().hasOneBlock())
      return rewriter.notifyMatchFailure(op, "multi-block loop unsupported");

    Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> newInductionVars;

    rewriter.setInsertionPoint(op);
    llvm::SmallVector<Value> lbs, steps, tripCounts;
    for (unsigned i = 0; i < op.getCollapseNumLoops(); i++) {
      Value lb = convertToIndexIfNot(op.getLoopLowerBounds()[i], loc, rewriter);
      Value ub = convertToIndexIfNot(op.getLoopUpperBounds()[i], loc, rewriter);
      Value step = convertToIndexIfNot(op.getLoopSteps()[i], loc, rewriter);
      // tripCount = (ub - lb + step)/step
      Value tc = arith::DivSIOp::create(
          rewriter, loc,
          arith::AddIOp::create(rewriter, loc,
                                arith::SubIOp::create(rewriter, loc, ub, lb),
                                step),
          step);
      lbs.push_back(lb);
      steps.push_back(step);
      tripCounts.push_back(tc);
    }

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    llvm::SmallVector<Value> zeros(tripCounts.size(), zero);
    llvm::SmallVector<Value> ones(tripCounts.size(), one);
    auto scfParallelOp =
        scf::ParallelOp::create(rewriter, loc, zeros, tripCounts, ones);

    // Move Fir Loop to ScfParallelOp
    rewriter.setInsertionPointToStart(scfParallelOp.getBody());
    for (auto [scfIv, lb, step, origIv] :
         llvm::zip(scfParallelOp.getInductionVars(), lbs, steps,
                   op.getRegion().front().getArguments())) {
      // Affine transformation to recover iv of fir:
      // fir: (lb, ub, step)
      // scf: (0, tc, 1)
      // so: iv = lb + i * step
      Value iv = arith::AddIOp::create(
          rewriter, loc, lb, arith::MulIOp::create(rewriter, loc, scfIv, step));
      if (iv.getType() != origIv.getType()) {
        iv = fir::ConvertOp::create(rewriter, loc, origIv.getType(), iv);
      }
      newInductionVars.push_back(iv);
    }

    auto *targetBlock = scfParallelOp.getBody();
    auto *sourceBlock = &op.getRegion().front();
    rewriter.eraseOp(sourceBlock->getTerminator());
    rewriter.inlineBlockBefore(sourceBlock, targetBlock,
                               targetBlock->getTerminator()->getIterator(),
                               newInductionVars);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult matchAndRewrite(omp::LoopNestOp op,
                                PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> wrappers = collectWrappers(op);
    if (llvm::any_of(wrappers, [](Operation *w) {
          return llvm::isa<omp::LoopNestOp>(w);
        })) {
      return rewriter.notifyMatchFailure(op,
                                         "outer loop_nest not flattened yet");
    }
    // According to OpenMP Dialect, the parent of LoopNestOp must be another
    // wrapper, all with parallelism semantics.
    // TODO: check if there is a clause prevent parallelism
    if (replaceNestLoopOp(op, rewriter).failed()) {
      return failure();
    }
    for (auto *wrapper : wrappers) {
      materializePrivates(wrapper, rewriter);
      materializeReductions(wrapper, rewriter);
      promoteAndDelete(wrapper, rewriter);
    }
    return success();
  }
};

class FlattenTargetPass
    : public flangomp::impl::FlattenOpenMPTargetBase<FlattenTargetPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp->getContext();
    OpBuilder opBuilder(ctx);

    // FIXME: use scf::ParallelOp instead of fir::DoLoop for wsLoopOp
    // Also, for scf::ParallelOp, use scf::ReduceOp instead of non-atomic
    // operations
    RewritePatternSet patterns(ctx);
    patterns.add<FlattenPattern>(ctx);
    patterns.add<MapHostEvalPattern>(ctx);
    GreedyRewriteConfig config;
    config.enableFolding();

    FrozenRewritePatternSet frozenRewritePatternSet(std::move(patterns));
    if (failed(
            applyPatternsGreedily(moduleOp, frozenRewritePatternSet, config))) {
      signalPassFailure();
      return;
    }
    return;
  };
};

} // namespace
