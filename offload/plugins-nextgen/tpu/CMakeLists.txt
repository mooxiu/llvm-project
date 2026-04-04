# Create the library and add the default arguments.
add_target_library(omptarget.rtl.tpu TPU)

target_sources(omptarget.rtl.tpu PRIVATE src/rtl.cpp)
target_include_directories(omptarget.rtl.tpu PRIVATE dynamic_tpu)

