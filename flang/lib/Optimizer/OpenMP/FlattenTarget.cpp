#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <mlir/Dialect/OpenMP/OpenMPDialect.h>
#include <mlir/Dialect/OpenMP/OpenMPOpsEnums.h>
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

  static unsigned getBlockHostEvalVarsOffset(omp::TargetOp targetOp) {
    return targetOp.numHasDeviceAddrBlockArgs();
  }

  static unsigned getBlockMapVarsOffset(omp::TargetOp targetOp) {
    return targetOp.numHasDeviceAddrBlockArgs()
      + targetOp.numInReductionBlockArgs()
      + targetOp.numHostEvalBlockArgs();
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
  

int getPrivateOffset(Operation* ompConstruct) {
  return 0;
}


// INFO: Only support private and firstprivate of scalar value.
// template<typename OMPConstruct> 
struct MaterializePrivatePattern: public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  omp::PrivateClauseOp findPrivateRecipe(omp::ParallelOp op, SymbolRefAttr privateSym) const {
    return SymbolTable::lookupNearestSymbolFrom<omp::PrivateClauseOp>(op, privateSym);
  }

  void materializePrivate(
    omp::DataSharingClauseType kind, 
    Type dataType,
    Value privateVar,
    Value arg
  ) const {

  }

  LogicalResult matchAndRewrite(omp::ParallelOp op, PatternRewriter &rewriter) const {
    auto offset = getPrivateOffset(op);
    if (op.numPrivateBlockArgs() == 0) {
      return failure();
    }

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
      materializePrivate(kind, type, privateVar, arg);
    }

    for (int i = 0; i < op.numPrivateBlockArgs(); i++) {
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

};

struct MapHostEvalPattern: public OpRewritePattern<omp::TargetOp> {
  using OpRewritePattern<omp::TargetOp>::OpRewritePattern;

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



  int getHostEvalOffset(omp::TargetOp op) const {
    // FIXME: IMPLEMENT ME!
    return 0;
  }

  LogicalResult matchAndRewrite(omp::TargetOp op, PatternRewriter &rewriter) const {
    auto offset = getHostEvalOffset(op);
    for (int i = 0; i < op.numHostEvalBlockArgs(); i++) {
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



