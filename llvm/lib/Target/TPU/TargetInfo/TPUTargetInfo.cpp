#include "TPUTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheTPUTarget() {
  static Target TheTPUTarget;
  return TheTPUTarget;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeTPUTargetInfo() {
  RegisterTarget<Triple::tpu, /*HasJIT=*/false> X(getTheTPUTarget(), "tpu",
                                                 "TPU", "TPU");
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeTPUTargetMC() {
}
