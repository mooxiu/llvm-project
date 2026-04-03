add_llvm_component_library(LLVMTPUInfo
  TPUTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  TPU
  )
