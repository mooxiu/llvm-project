//===----RTLs/tpu/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for TPU machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <string>
#include "../dynamic_tpu/pjrt_c_api.h"
#include <unordered_map>

#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/Environment.h"

#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "Utils/ELF.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

using namespace error;

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct TPUKernelTy;
struct TPUDeviceTy;
struct TPUPluginTy;

struct TPUDeviceImageTy : public DeviceImageTy {
};

/// Class implementing the CUDA kernel functionalities which derives from the
/// generic kernel class.
struct TPUKernelTy : public GenericKernelTy {
  TPUKernelTy(const char *Name) : GenericKernelTy(Name) {}

  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    llvm_unreachable("TPU Kernel Ty, initImpl!");
    return Plugin::success();
  }

  Error
  delegatedLaunchImpl(GenericDeviceTy &GenericDevice,
                      std::function<int64_t(void *)> &DelegatedLaunch,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  /// Return maximum block size for maximum occupancy
  Expected<uint64_t> maxGroupSize(GenericDeviceTy &,
                                  uint64_t DynamicMemSize) const override {
    llvm_unreachable("TPU Kernel Ty, maxGroupSize!");
  }
};

struct TPUStreamRef final : public GenericDeviceResourceRef {
  /// The underlying handle type for streams.
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override {
    llvm_unreachable("StreamRef create!");
    return Plugin::success();
  }

  /// Destroy the referenced stream and invalidate the reference. The reference
  /// must be to a valid stream before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    llvm_unreachable("StreamRef destroy!");
    return Plugin::success();
  }
};

struct TPUEventRef final : public GenericDeviceResourceRef {
  /// Create a reference to an existing event.
  /// Create a new event and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override {
    llvm_unreachable("EventRef create");
    return Plugin::success();
  }

  /// Destroy the referenced event and invalidate the reference. The reference
  /// must be to a valid event before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    llvm_unreachable("EventRef destroy");
    return Plugin::success();
  }

};

struct TPUDeviceTy : public GenericDeviceTy {
  TPUDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, NVPTXGridValues) {}

  ~TPUDeviceTy() {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    llvm_unreachable("TPUDeviceTy initImpl");
    return Plugin::success();
  }

  Error unloadBinaryImpl(DeviceImageTy *Image) override {
    llvm_unreachable("TPUDeviceTy unloadBinaryImpl");
    return Plugin::success();
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override {
    llvm_unreachable("TPUDeviceTy deinitImpl");
    return Plugin::success();
  }

  virtual Error callGlobalConstructors(GenericPluginTy &Plugin,
                                       DeviceImageTy &Image) override {
    llvm_unreachable("TPUDeviceTy callGlobalConstructors");
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
  }

  virtual Error callGlobalDestructors(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) override {
    llvm_unreachable("TPUDeviceTy callGlobalDestructors");
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
  }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const {
    llvm_unreachable("TPUDeviceTy doJITPostProcessing");
    // TODO: We should be able to use the 'nvidia-ptxjitcompiler' interface to
    //       avoid the call to 'ptxas'.
    SmallString<128> PTXInputFilePath;
    std::error_code EC = sys::fs::createTemporaryFile("nvptx-pre-link-jit", "s",
                                                      PTXInputFilePath);
    if (EC)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to create temporary file for ptxas");

    // Write the file's contents to the output file.
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(PTXInputFilePath, MB->getBuffer().size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    llvm::copy(MB->getBuffer(), Output->getBufferStart());
    if (Error E = Output->commit())
      return std::move(E);

    SmallString<128> PTXOutputFilePath;
    EC = sys::fs::createTemporaryFile("nvptx-post-link-jit", "cubin",
                                      PTXOutputFilePath);
    if (EC)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to create temporary file for ptxas");

    // Try to find `ptxas` in the path to compile the PTX to a binary.
    const auto ErrorOrPath = sys::findProgramByName("ptxas");
    if (!ErrorOrPath)
      return Plugin::error(ErrorCode::HOST_TOOL_NOT_FOUND,
                           "failed to find 'ptxas' on the PATH.");

    std::string Arch = getComputeUnitKind();
    StringRef Args[] = {*ErrorOrPath,
                        "-m64",
                        "-O2",
                        "--gpu-name",
                        Arch,
                        "--output-file",
                        PTXOutputFilePath,
                        PTXInputFilePath};

    std::string ErrMsg;
    if (sys::ExecuteAndWait(*ErrorOrPath, Args, std::nullopt, {}, 0, 0,
                            &ErrMsg))
      return Plugin::error(ErrorCode::ASSEMBLE_FAILURE,
                           "running 'ptxas' failed: %s\n", ErrMsg.c_str());

    auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(PTXOutputFilePath.data());
    if (!BufferOrErr)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to open temporary file for ptxas");

    // Clean up the temporary files afterwards.
    if (sys::fs::remove(PTXOutputFilePath))
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to remove temporary file for ptxas");
    if (sys::fs::remove(PTXInputFilePath))
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to remove temporary file for ptxas");

    return std::move(*BufferOrErr);
  }

  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    llvm_unreachable("TPUDeviceTy constructKernel");
    TPUKernelTy *TPUKernel = Plugin.allocate<TPUKernelTy>();
    if (!TPUKernel)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "failed to allocate memory for TPU kernel");

    new (TPUKernel) TPUKernelTy(Name);

    return *TPUKernel;
  }

  /// Set the current context to this device's context.
  Error setContext() override {
    llvm_unreachable("TPUDeviceTy setContext");
    return Plugin::success();
  }

  /// NVIDIA returns the product of the SM count and the number of warps that
  /// fit if the maximum number of threads were scheduled on each SM.
  uint64_t getHardwareParallelism() const override {
    llvm_unreachable("TPUDeviceTy getHardwareParallelism");
    return HardwareParallelism;
  }

  /// We want to set up the RPC server for host services to the GPU if it is
  /// available.
  bool shouldSetupRPCServer() const override { 
    llvm_unreachable("TPUDeviceTy shouldSetupRPCServer");
    return true; }

  /// The RPC interface should have enough space for all available parallelism.
  uint64_t requestedRPCPortCount() const override {
    llvm_unreachable("TPUDeviceTy requestedRPCPortCount");
    return getHardwareParallelism();
  }

  /// Allocate memory on the device or related to the device.
  Expected<void *> allocate(size_t Size, void *, TargetAllocTy Kind) override {
    llvm_unreachable("TPUDeviceTy allocate");
    return nullptr;
  }

  /// Deallocate memory on the device or related to the device.
  Error free(void *TgtPtr, TargetAllocTy Kind) override {
    llvm_unreachable("TPUDeviceTy free");
    return Plugin::success();
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override {
    llvm_unreachable("TPUDeviceTy synchronizeImpl");
    return Plugin::success();
  }

  /// Allocates \p RSize bytes (rounded up to page size) and hints the cuda
  /// driver to map it to \p VAddr. The obtained address is stored in \p Addr.
  /// At return \p RSize contains the actual size
  Error memoryVAMap(void **Addr, void *VAddr, size_t *RSize) override {
    llvm_unreachable("TPUDeviceTy memoryVAMap");
    return Plugin::success();
  }

  /// De-allocates device memory and Unmaps the Virtual Addr
  Error memoryVAUnMap(void *VAddr, size_t Size) override {
    llvm_unreachable("TPUDeviceTy memoryVAUnMap");
    return Plugin::success();
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    llvm_unreachable("TPUDeviceTy queryAsyncImpl");
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    llvm_unreachable("TPUDeviceTy dataLockImpl");
  }

  Error dataUnlockImpl(void *HstPtr) override {
    llvm_unreachable("TPUDeviceTy dataUnlockImpl");
    return Plugin::success(); }

  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    llvm_unreachable("TPUDeviceTy isPinnedPtrImpl");
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("TPUDeviceTy dataSubmitImpl");
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("TPUDeviceTy dataRetrieveImpl");
   }

  /// Exchange data between two devices directly. We may use peer access if
  /// the CUDA devices and driver allow them.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  Error dataFillImpl(void *TgtPtr, const void *PatternPtr, int64_t PatternSize,
                     int64_t Size,
                     AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("TPUDeviceTy  dataFillImpl");
  }

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("TPUDeviceTy initAsyncInfoImpl");
  }

  /// Insert a data fence between previous data operations and the following
  /// operations. This is a no-op for CUDA devices as operations inserted into
  /// a queue are in-order.
  Error dataFence(__tgt_async_info *Async) override {
    llvm_unreachable("TPUDeviceTy dataFence");
    return Plugin::success();
  }

  interop_spec_t selectInteropPreference(int32_t InteropType,
                                         int32_t NumPrefers,
                                         interop_spec_t *Prefers) override {
    llvm_unreachable("TPUDeviceTy selectInteropPreference");
    return interop_spec_t{tgt_fr_cuda, {true, 0}, 0};
  }

  Expected<omp_interop_val_t *>
  createInterop(int32_t InteropType, interop_spec_t &InteropSpec) override {
    llvm_unreachable("TPUDeviceTy createInterop");
  }

  Error releaseInterop(omp_interop_val_t *Interop) override {
    llvm_unreachable("TPUDeviceTy releaseInterop");
    if (!Interop)
      return Plugin::success();

    if (Interop->async_info)
      delete Interop->async_info;

    delete Interop;
    return Plugin::success();
  }

  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &AsyncInfo) override {
    llvm_unreachable("TPUDeviceTy enqueueHostCallImpl");
  };

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override {
    llvm_unreachable("TPUDeviceTy createEventImpl");
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("TPUDeviceTy waitEventImpl");
  }

  Expected<bool> isEventCompleteImpl(void *EventPtr,
                                     AsyncInfoWrapperTy &) override {
    llvm_unreachable("TPUDeviceTy isEventCompleteImpl");
  }

  /// Print information about the device.
  Expected<InfoTreeNode> obtainInfoImpl() override {
    llvm_unreachable("TPUDeviceTy obtainInfoImpl");
  }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    llvm_unreachable("TPUDeviceTy getDeviceStackSize");
  }
  Error setDeviceStackSize(uint64_t Value) override {
    llvm_unreachable("TPUDeviceTy setDeviceStackSize");
  }
  bool hasDeviceHeapSize() override { 
    return true; 
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    llvm_unreachable("TPUDeviceTy getDeviceHeapSize");
  }
  Error setDeviceHeapSize(uint64_t Value) override {
    llvm_unreachable("TPUDeviceTy setDeviceHeapSize");
  }
  Error getDeviceMemorySize(uint64_t &Value) override {
    llvm_unreachable("TPUDeviceTy getDeviceMemorySize");
  }

  Error getDeviceAttr(uint32_t Kind, uint32_t &Value) {
    llvm_unreachable("TPUDeviceTy getDeviceAttr");
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override {
    llvm_unreachable("TPUDeviceTy getComputeUnitKind");
  }

  /// Returns the clock frequency for the given NVPTX device.
  uint64_t getClockFrequency() const override { 
    llvm_unreachable("TPUDeviceTy getClockFrequency");
    return 1000000000; }

  Error callGlobalCtorDtorCommon(GenericPluginTy &Plugin, DeviceImageTy &Image,
                                 bool IsCtor) {
    llvm_unreachable("TPUDeviceTy callGlobalCtorDtorCommon");
  }

  struct ComputeCapabilityTy {
    uint32_t Major;
    uint32_t Minor;
    std::string str() const {
      return "sm_" + std::to_string(Major * 10 + Minor);
    }
  } ComputeCapability;

  /// The maximum number of warps that can be resident on all the SMs
  /// simultaneously.
  uint32_t HardwareParallelism = 0;
};

Error TPUKernelTy::delegatedLaunchImpl(
    GenericDeviceTy &GenericDevice,
    std::function<int64_t(void *)> &DelegatedLaunch,
    AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  llvm_unreachable("delegatedLaunchImpl");
}

Error TPUKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                               uint32_t NumThreads[3], uint32_t NumBlocks[3],
                               KernelArgsTy &KernelArgs,
                               KernelLaunchParamsTy LaunchParams,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  llvm_unreachable("TPUKernelTy::launchImpl");
}

class TPUGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    llvm_unreachable("getGlobalMetadataFromDevice");
    return Plugin::success();
  }
};

struct TPUPluginTy final : public GenericPluginTy {
  TPUPluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  TPUPluginTy(const TPUPluginTy &) = delete;
  TPUPluginTy(TPUPluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    llvm_unreachable("initImpl");
    return 1;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    llvm_unreachable("deinitImpl");
    return Plugin::success(); }

  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    llvm_unreachable("createDevice");
  }

  GenericGlobalHandlerTy *createGlobalHandler() override {
    llvm_unreachable("createGlobalHandler");
    return new TPUGlobalHandlerTy();
  }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { 
    llvm_unreachable("getMagicElfBits");
  }

  Triple::ArchType getTripleArch() const override {
    return Triple::tpu;
  }

  const char *getName() const override { 
    llvm_unreachable("getName");
    return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    llvm_unreachable("isELFCompatible");
  }
};

Error TPUDeviceTy::dataExchangeImpl(const void *SrcPtr,
                                     GenericDeviceTy &DstGenericDevice,
                                     void *DstPtr, int64_t Size,
                                     AsyncInfoWrapperTy &AsyncInfoWrapper) {
  llvm_unreachable("TPUDeviceTy::dataExchangeImpl");
}

template <typename... ArgsTy>
static Error Plugin::check(int32_t Code, const char *ErrFmt, ArgsTy... Args) {
  llvm_unreachable("Plugin::check");
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_tpu() {
  return new llvm::omp::target::plugin::TPUPluginTy();
}
}
