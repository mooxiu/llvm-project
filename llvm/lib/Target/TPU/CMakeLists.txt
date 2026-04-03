add_llvm_component_group(TPU)

add_llvm_component_library(LLVMTPUCodeGen
  TPUTargetMachine.cpp

  LINK_COMPONENTS
  Core
  Support
  Target
  TargetParser
  TPUInfo

  ADD_TO_COMPONENT
  TPU
)

add_subdirectory(TargetInfo)
