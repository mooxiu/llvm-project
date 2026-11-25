// Regressional Test: flang/test/Transforms/OpenMP/flatten-target.mlir

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
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
#include <mlir/IR/IRMapping.h>
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

static omp::PrivateClauseOp getPrivateRecipe(Operation *wrapper,
                                             unsigned index) {
  auto syms = wrapper->getAttrOfType<ArrayAttr>("private_syms");
  assert(syms && index < syms.size());
  return SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(
      wrapper, cast<SymbolRefAttr>(syms[index]));
}

static omp::DeclareReductionOp getReductionRecipe(Operation *wrapper,
                                                  unsigned index) {
  auto syms = wrapper->getAttrOfType<ArrayAttr>("reduction_syms");
  assert(syms && index < syms.size());
  return SymbolTable::lookupNearestSymbolFrom<omp::DeclareReductionOp>(
      wrapper, cast<SymbolRefAttr>(syms[index]));
}

// Clones the single block of `region` at the insertion point, with its
// arguments replaced by `args`, and returns the value it yields.
static Value cloneRegionInline(Region &region, ValueRange args,
                               PatternRewriter &rewriter) {
  Block &block = region.front();
  IRMapping mapping;
  for (auto [from, to] : llvm::zip(block.getArguments(), args)) {
    mapping.map(from, to);
  }
  for (Operation &op : block.without_terminator()) {
    rewriter.clone(op, mapping);
  }
  return mapping.lookupOrDefault(block.getTerminator()->getOperand(0));
}

namespace {
// One item of a reduction clause on a wrapper of the loop.
struct ReductionItem {
  Operation *wrapper;
  unsigned index;
  omp::DeclareReductionOp recipe;
  // True if the reduced variable is defined outside all wrappers: the loop
  // reduces into it with scf.reduce. Otherwise it is the copy of an enclosing
  // wrapper's reduction (e.g. `teams parallel do reduction(+:s)` gives teams
  // and wsloop a reduction each, the wsloop one on the teams copy). That copy
  // is per iteration too, so the inner copy is combined into it at the end of
  // the iteration.
  bool onLoop;
  // The per-iteration copy, set when it is built.
  Value copy;
};
} // namespace

// Checks that the private and reduction variables of `wrappers` can get one
// copy per iteration of the parallel loop, and collects the ops that must move
// into the loop body with them.
//
// Every private or reduction block argument is replaced by an alloca inside
// the loop body. Its users outside the loop (e.g. the hlfir.declare of the
// copy in an omp.parallel block) would then use a value that does not
// dominate them, so they move into the loop body too. They used to run once
// per thread and will run once per iteration, which is only equivalent if they
// have no side effects.
static LogicalResult analyzeWrappers(omp::LoopNestOp loop,
                                     ArrayRef<Operation *> wrappers,
                                     llvm::SetVector<Operation *> &toMove,
                                     llvm::SmallVectorImpl<ReductionItem> &reductions) {
  auto isScalar = [](Type type) {
    return type.isIntOrIndexOrFloat() || isa<fir::LogicalType>(type);
  };
  llvm::SmallVector<Value> worklist;
  // Outer wrappers first, so `reductions` lists enclosing reductions first.
  for (Operation *wrapper : llvm::reverse(wrappers)) {
    auto iface = dyn_cast<omp::BlockArgOpenMPOpInterface>(wrapper);
    if (!iface) {
      return wrapper->emitError("unsupported OpenMP loop wrapper");
    }
    auto byref = wrapper->getAttrOfType<DenseBoolArrayAttr>("reduction_byref");
    for (auto [i, arg] : llvm::enumerate(iface.getReductionBlockArgs())) {
      omp::DeclareReductionOp recipe = getReductionRecipe(wrapper, i);
      // Only by-value scalars for now: the copy is a plain alloca set to the
      // init region's value, and the init region must not read its argument
      // (the original value, only needed as a mold for allocations).
      Region &init = recipe.getInitializerRegion();
      if ((byref && byref.asArrayRef()[i]) || !recipe.getAllocRegion().empty() ||
          !recipe.getCleanupRegion().empty() || !init.hasOneBlock() ||
          !init.front().getArgument(0).use_empty() ||
          !recipe.getReductionRegion().hasOneBlock() ||
          !isScalar(recipe.getType())) {
        return recipe.emitError(
            "only by-value scalar reductions are supported");
      }
      Value var = iface.getReductionVars()[i];
      bool onLoop = !wrappers.back()->isAncestor(
          var.getParentRegion()->getParentOp());
      reductions.push_back({wrapper, static_cast<unsigned>(i), recipe, onLoop,
                            Value()});
      worklist.push_back(arg);
    }
    for (auto [i, arg] : llvm::enumerate(iface.getPrivateBlockArgs())) {
      omp::PrivateClauseOp recipe = getPrivateRecipe(wrapper, i);
      Type type = recipe.getType();
      // Only scalars for now: their copy is a plain alloca (+ load/store for
      // firstprivate). Others need the init/copy/dealloc regions.
      if (!recipe.getInitRegion().empty() ||
          !recipe.getDeallocRegion().empty() || !isScalar(type)) {
        return recipe.emitError("only scalar private variables are supported");
      }
      worklist.push_back(arg);
    }
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (loop->isAncestor(user) || toMove.contains(user)) {
        continue;
      }
      // A nested wrapper may privatize or reduce this value again; that
      // wrapper's copy is built inside the loop body.
      if (llvm::is_contained(wrappers, user)) {
        auto iface = cast<omp::BlockArgOpenMPOpInterface>(user);
        if (!llvm::is_contained(iface.getPrivateVars(), value) &&
            !llvm::is_contained(iface.getReductionVars(), value)) {
          return user->emitError("private or reduction variable used by a "
                                 "clause other than private or reduction");
        }
        continue;
      }
      Operation *loopInBlock =
          user->getBlock()->findAncestorOpInBlock(*loop.getOperation());
      bool beforeLoop = llvm::is_contained(wrappers, user->getParentOp()) &&
                        loopInBlock && user->isBeforeInBlock(loopInBlock);
      bool pure = isa<hlfir::DeclareOp, fir::DeclareOp>(user) ||
                  (user->getNumRegions() == 0 && isMemoryEffectFree(user));
      if (!beforeLoop || !pure) {
        return user->emitError("cannot move this use of a private or "
                               "reduction variable into the loop");
      }
      toMove.insert(user);
      for (Value result : user->getResults()) {
        worklist.push_back(result);
      }
    }
  }

  // A reduction that is not on the loop combines into an enclosing copy at
  // the end of each iteration, so its variable must be that copy (or a value
  // moved into the loop with it), not something shared by all iterations.
  for (ReductionItem &r : reductions) {
    if (r.onLoop) {
      continue;
    }
    Value var =
        cast<omp::BlockArgOpenMPOpInterface>(r.wrapper).getReductionVars()
            [r.index];
    auto arg = dyn_cast<BlockArgument>(var);
    bool isEnclosingCopy =
        (arg && llvm::is_contained(wrappers, arg.getOwner()->getParentOp())) ||
        toMove.contains(var.getDefiningOp());
    if (!isEnclosingCopy) {
      return r.wrapper->emitError(
          "reduction variable must be defined outside the loop's constructs "
          "or be the copy of an enclosing reduction or private variable");
    }
  }
  return success();
}

// Replaces each private block argument by a fresh copy created at the current
// insertion point, which the caller puts in the loop body so that every
// iteration gets its own copy. `private` copies start uninitialized;
// `firstprivate` copies start with the value of the original variable.
template <typename T>
static void materializePrivatesImpl(T op, PatternRewriter &rewriter) {
  auto iface = dyn_cast<omp::BlockArgOpenMPOpInterface>(op.getOperation());
  if (!iface)
    return;

  auto offset = iface.getPrivateBlockArgsStart();
  if (iface.numPrivateBlockArgs() == 0) {
    return;
  }

  auto privateVarNum = iface.numPrivateBlockArgs();
  auto loc = iface->getLoc();

  for (unsigned i = 0; i < privateVarNum; i++) {
    auto privateVar = iface.getPrivateVars()[i];
    auto arg = iface.getPrivateBlockArgs()[i];
    omp::PrivateClauseOp recipe = getPrivateRecipe(op.getOperation(), i);

    auto local = fir::AllocaOp::create(rewriter, loc, recipe.getType());
    if (recipe.getDataSharingType() ==
        omp::DataSharingClauseType::FirstPrivate) {
      auto loadOp = fir::LoadOp::create(rewriter, loc, privateVar);
      fir::StoreOp::create(rewriter, loc, loadOp.getResult(),
                           local.getResult());
    }
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

// Replaces the reduction block argument of `r` by a fresh copy created at the
// current insertion point (in the loop body, so one per iteration), starting
// at the identity value given by the recipe's init region.
static void materializeReductionCopy(ReductionItem &r, PatternRewriter &rewriter) {
  auto iface = cast<omp::BlockArgOpenMPOpInterface>(r.wrapper);
  Location loc = r.wrapper->getLoc();
  auto copy = fir::AllocaOp::create(rewriter, loc, r.recipe.getType());
  Value identity =
      cloneRegionInline(r.recipe.getInitializerRegion(), ValueRange(),
                        rewriter);
  fir::StoreOp::create(rewriter, loc, identity, copy.getResult());
  rewriter.replaceAllUsesWith(iface.getReductionBlockArgs()[r.index],
                              copy.getResult());
  r.copy = copy.getResult();
}

// Removes the reduction clause; its block arguments have no uses left.
template <typename T>
static void eraseReductionClauseImpl(T op, PatternRewriter &rewriter) {
  auto iface = cast<omp::BlockArgOpenMPOpInterface>(op.getOperation());
  unsigned start = iface.getReductionBlockArgsStart();
  unsigned num = iface.numReductionBlockArgs();
  if (num == 0) {
    return;
  }
  rewriter.modifyOpInPlace(op, [&] {
    op.getReductionVarsMutable().clear();
    op->removeAttr("reduction_syms");
    op->removeAttr("reduction_byref");
    op->removeAttr("reduction_mod");
    op.getRegion().front().eraseArguments(start, num);
  });
}

static void eraseReductionClause(Operation *op, PatternRewriter &rewriter) {
  llvm::TypeSwitch<Operation *>(op)
      .Case<omp::ParallelOp, omp::TeamsOp, omp::WsloopOp, omp::LoopOp,
            omp::SimdOp>([&](auto typedOp) {
        return eraseReductionClauseImpl(typedOp, rewriter);
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

  // Replaces `op` by an scf.parallel. Each of `reduceVars` is loaded before
  // the loop as an init value of the scf.parallel; the caller ends the body
  // with the matching scf.reduce and stores the results back.
  FailureOr<scf::ParallelOp> replaceNestLoopOp(omp::LoopNestOp op,
                                               ValueRange reduceVars,
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
    llvm::SmallVector<Value> inits;
    for (Value var : reduceVars) {
      inits.push_back(fir::LoadOp::create(rewriter, loc, var));
    }
    auto scfParallelOp = scf::ParallelOp::create(rewriter, loc, zeros,
                                                 tripCounts, ones, inits);

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

    // Without init values the builder adds an empty scf.reduce terminator;
    // with them the body has no terminator yet.
    auto *targetBlock = scfParallelOp.getBody();
    auto *sourceBlock = &op.getRegion().front();
    Block::iterator insertPt = inits.empty()
                                   ? targetBlock->getTerminator()->getIterator()
                                   : targetBlock->end();
    rewriter.eraseOp(sourceBlock->getTerminator());
    rewriter.inlineBlockBefore(sourceBlock, targetBlock, insertPt,
                               newInductionVars);
    rewriter.eraseOp(op);
    return scfParallelOp;
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
    llvm::SetVector<Operation *> toMove;
    llvm::SmallVector<ReductionItem> reductions;
    if (failed(analyzeWrappers(op, wrappers, toMove, reductions))) {
      return failure();
    }
    llvm::SmallVector<ReductionItem *> loopReductions;
    llvm::SmallVector<Value> loopReduceVars;
    for (ReductionItem &r : reductions) {
      if (r.onLoop) {
        loopReductions.push_back(&r);
        loopReduceVars.push_back(
            cast<omp::BlockArgOpenMPOpInterface>(r.wrapper)
                .getReductionVars()[r.index]);
      }
    }
    FailureOr<scf::ParallelOp> loop =
        replaceNestLoopOp(op, loopReduceVars, rewriter);
    if (failed(loop)) {
      return failure();
    }
    Location loc = loop->getLoc();
    Block *body = loop->getBody();

    // Build the private and reduction copies inside the loop body, outer
    // wrappers first, and move the ops that use them in after them. The body
    // then starts with
    //   [outer copies][outer moved ops][inner copies][inner moved ops][iv]...
    // so every copy and moved op comes after the values it uses.
    Operation *bodyStart = &body->front();
    for (Operation *wrapper : llvm::reverse(wrappers)) {
      rewriter.setInsertionPoint(bodyStart);
      materializePrivates(wrapper, rewriter);
      for (ReductionItem &r : reductions) {
        if (r.wrapper == wrapper) {
          materializeReductionCopy(r, rewriter);
        }
      }
      for (Operation &inner :
           llvm::make_early_inc_range(wrapper->getRegion(0).front())) {
        if (toMove.contains(&inner)) {
          rewriter.moveOpBefore(&inner, bodyStart);
        }
      }
    }

    // End of each iteration. First combine every copy that belongs to an
    // enclosing copy into it, inner ones first, then reduce the copies of the
    // variables outside the loop with scf.reduce.
    if (body->mightHaveTerminator()) {
      rewriter.setInsertionPoint(body->getTerminator());
    } else {
      rewriter.setInsertionPointToEnd(body);
    }
    for (ReductionItem &r : llvm::reverse(reductions)) {
      if (r.onLoop) {
        continue;
      }
      // Read the operand now: it was replaced by the enclosing copy.
      Value var = cast<omp::BlockArgOpenMPOpInterface>(r.wrapper)
                      .getReductionVars()[r.index];
      Value outer = fir::LoadOp::create(rewriter, loc, var);
      Value local = fir::LoadOp::create(rewriter, loc, r.copy);
      Value combined = cloneRegionInline(r.recipe.getReductionRegion(),
                                         {outer, local}, rewriter);
      fir::StoreOp::create(rewriter, loc, combined, var);
    }
    if (!loopReductions.empty()) {
      llvm::SmallVector<Value> contributions;
      for (ReductionItem *r : loopReductions) {
        contributions.push_back(fir::LoadOp::create(rewriter, loc, r->copy));
      }
      auto reduce = scf::ReduceOp::create(rewriter, loc, contributions);
      for (auto [region, r] : llvm::zip(reduce.getReductions(), loopReductions)) {
        Block &block = region.front();
        rewriter.setInsertionPointToStart(&block);
        Value combined = cloneRegionInline(r->recipe.getReductionRegion(),
                                           block.getArguments(), rewriter);
        scf::ReduceReturnOp::create(rewriter, loc, combined);
      }
      // The init values were the original variables, so the results already
      // include them.
      rewriter.setInsertionPointAfter(*loop);
      for (auto [result, var] : llvm::zip(loop->getResults(), loopReduceVars)) {
        fir::StoreOp::create(rewriter, loc, result, var);
      }
    }

    for (Operation *wrapper : wrappers) {
      eraseReductionClause(wrapper, rewriter);
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
    // FlattenPattern reports why it rejected a loop; do not let the pass
    // succeed with OpenMP loops left in the IR.
    WalkResult result = moduleOp.walk([](omp::LoopNestOp loop) {
      loop.emitError("OpenMP loop was not flattened");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
    return;
  };
};

} // namespace
