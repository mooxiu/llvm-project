# Create the library and add the default arguments.
add_target_library(omptarget.rtl.vpu VirPU)

target_sources(omptarget.rtl.vpu PRIVATE src/rtl.cpp)
target_include_directories(omptarget.rtl.vpu PRIVATE dynamic_vpu)

