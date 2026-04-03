//===-- TPUTargetMachine.h - Define TargetMachine for TPU ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TPU specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "TPUSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <optional>
#include <utility>

namespace llvm {

/// TPUTargetMachine
///
class TPUTargetMachine : public CodeGenTargetMachineImpl {
  bool is64bit;
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  TPU::DrvInterface drvInterface;
  TPUSubtarget Subtarget;

  // Hold Strings that can be free'd all together with TPUTargetMachine
  BumpPtrAllocator StrAlloc;
  UniqueStringSaver StrPool;

public:
  TPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     std::optional<Reloc::Model> RM,
                     std::optional<CodeModel::Model> CM, CodeGenOptLevel OP,
                     bool is64bit);
  ~TPUTargetMachine() override;
  const TPUSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }
  const TPUSubtarget *getSubtargetImpl() const { return &Subtarget; }
  bool is64Bit() const { return is64bit; }
  TPU::DrvInterface getDrvInterface() const { return drvInterface; }
  UniqueStringSaver &getStrPool() const {
    return const_cast<UniqueStringSaver &>(StrPool);
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  // Emission of machine code through MCJIT is not supported.
  bool addPassesToEmitMC(PassManagerBase &, MCContext *&, raw_pwrite_stream &,
                         bool = true) override {
    return true;
  }
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  void registerEarlyDefaultAliasAnalyses(AAManager &AAM) override;

  void registerPassBuilderCallbacks(PassBuilder &PB) override;

  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;

  bool isMachineVerifierClean() const override {
    return false;
  }

  std::pair<const Value *, unsigned>
  getPredicatedAddrSpace(const Value *V) const override;
}; // TPUTargetMachine.


} // end namespace llvm

