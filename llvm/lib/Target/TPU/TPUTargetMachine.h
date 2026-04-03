#ifndef LLVM_LIB_TARGET_TPU_TPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_TPU_TPUTARGETMACHINE_H

#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class TPUTargetMachine : public CodeGenTargetMachineImpl {
public:
  TPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                   StringRef FS, const TargetOptions &Options,
                   std::optional<Reloc::Model> RM,
                   std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                   bool JIT);
};
} // namespace llvm

#endif
