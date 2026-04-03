add_llvm_component_library(LLVMTPUDesc

  LINK_COMPONENTS
  CodeGenTypes
  MC
  Support
  TargetParser
  TPUInfo

  ADD_TO_COMPONENT
  TPU
  )
