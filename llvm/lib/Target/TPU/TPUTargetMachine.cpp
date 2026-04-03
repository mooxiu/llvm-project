#include "TPUTargetMachine.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/MC/TargetRegistry.h"
#include "TargetInfo/TPUTargetInfo.h"

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeTPUTarget() {
  RegisterTargetMachine<TPUTargetMachine> X(getTheTPUTarget());
}

TPUTargetMachine::TPUTargetMachine(const Target &T, const Triple &TT,
                                   StringRef CPU, StringRef FS,
                                   const TargetOptions &Options,
                                   std::optional<Reloc::Model> RM,
                                   std::optional<CodeModel::Model> CM,
                                   CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(
          T,
          "e-m:e-p:64:64-i64:64-n32:64-S128", //TPU DataLayout
          TT, CPU, FS, Options, 
          RM.value_or(Reloc::PIC_),           // PIC (Position Independent Code)
          CM.value_or(CodeModel::Small),      // Small CodeModel
          OL) {}

