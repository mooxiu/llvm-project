#include "flang/Optimizer/OpenMP/Passes.h"

namespace flangomp {
  #define GEN_PASS_DEF_MATERIALIZEPRIVATES
  #include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

#define DEBUG_TYPE "materialize-privates"


using namespace mlir;

namespace {
  class MaterializePrivatesPass 
    : public flangomp::impl::MaterializePrivatesBase<MaterializePrivatesPass> {
  public:
    void runOnOperation() override {
       
    }
  };
}


