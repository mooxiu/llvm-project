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
#include <dlfcn.h>
#include <iostream>
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
#include "llvm/Support/Debug.h"
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
    return Plugin::success();
  }

  Error unloadBinaryImpl(DeviceImageTy *Image) override {
    return Plugin::success();
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override {
    return Plugin::success();
  }

  virtual Error callGlobalConstructors(GenericPluginTy &Plugin,
                                       DeviceImageTy &Image) override {
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
  }

  virtual Error callGlobalDestructors(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) override {
    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
  }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const override{
    llvm_unreachable("TPUDeviceTy doJITPostProcessing");
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
    DeviceGlobal.setPtr(nullptr);
    return Plugin::success();
  }
};

struct TPUPluginTy final : public GenericPluginTy {
  PJRT_Api PjrtApi;
  PJRT_Client* PjrtClient = nullptr;
  PJRT_Device* PjrtDevice = nullptr;

  TPUPluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  TPUPluginTy(const TPUPluginTy &) = delete;
  TPUPluginTy(TPUPluginTy &&) = delete;


  std::string getDeviceDescription(const PJRT_Api *api, PJRT_Device *device) {
    PJRT_Device_GetDescription_Args args = {
      .struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
      .device = device,
    };
    auto err1 = api->PJRT_Device_GetDescription(&args);
    if (err1) {
      std::cerr << "Error in getting description!\n";
      return nullptr;
    }
    PJRT_DeviceDescription_ToString_Args ts_args = {
      .struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE,
      .device_description = args.device_description,
    };
    auto err2 = api->PJRT_DeviceDescription_ToString(&ts_args);
    if (err2) {
      std::cerr << "Error in getting description to string!\n";
      return nullptr;
    }
    return ts_args.to_string;
  }

  // Get the target device handle
  PJRT_Device *findDevice(const PJRT_Api *api, PJRT_Client *client,
                          const std::string &deviceDescKeyword) {
    PJRT_Client_AddressableDevices_Args device_args = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
      .client = client,
    };
    auto err = api->PJRT_Client_AddressableDevices(&device_args);
    if (err || device_args.num_addressable_devices < 1) {
      std::cerr << "no devices found!\n"; 
      return nullptr;
    }

    int chosen_device_idx = -1;
    std::string desc = ""; // for logging purpose
    for (int i = 0; i < device_args.num_addressable_devices; i++) {
      std::string tmp = getDeviceDescription(api, device_args.addressable_devices[i]);
      llvm::dbgs() << "We're getting description like: " << tmp << "\n" ;
      std::transform(tmp.begin(), tmp.end(), tmp.begin(),[](unsigned char c) { return std::tolower(c); });
      if (tmp.find(deviceDescKeyword) != std::string::npos) {
        chosen_device_idx = i;
        desc = tmp;
        break;
      }
    }
    if (chosen_device_idx == -1) {
      std::cerr << "no device found, but why?!\n";
      return nullptr;
    }
    return device_args.addressable_devices[chosen_device_idx];
  }


  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    const char* custom_path = std::getenv("LIBTPU_PATH");
    void* Handle = nullptr;
    if (custom_path != nullptr) {
      Handle =  dlopen(custom_path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    } else {
      Handle =  dlopen("libtpu.so", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    }
    if (!Handle) {
      std::cerr << "error loading plugin: " << dlerror() << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // follow the example of `man dlopen`
    auto GetApiFn = (PJRT_Api * (*)()) dlsym(Handle, "GetPjrtApi");
    if (!GetApiFn) {
      std::cerr << "error finding GetPjrtApi: " << dlerror() << std::endl;
      std::exit(EXIT_FAILURE);
    }
    PJRT_Api* Api = GetApiFn();
    assert(Api && "Can not get APi!");
    PJRT_Plugin_Initialize_Args InitArgs = {};
    InitArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    auto* InitErr = Api->PJRT_Plugin_Initialize(&InitArgs);
    // Theoretically need to close handle_ when exiting, but it will automatically be destroyed when exiting the program so intentionally leave it.
    PJRT_Client_Create_Args args = {
      .struct_size= PJRT_Client_Create_Args_STRUCT_SIZE
    };
    auto* error = Api->PJRT_Client_Create(&args);
    if (error) {
      std::cerr << "Fail to create client!\n";
      std::exit(EXIT_FAILURE);
    }
    this->PjrtClient = args.client;
    auto device = findDevice(Api, PjrtClient, "cuda");
    this->PjrtDevice = device;
    // Should return number of devices
    //
    return 1;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    llvm_unreachable("deinitImpl");
    return Plugin::success(); }

  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new TPUDeviceTy(Plugin, DeviceId, NumDevices);
  }

  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new TPUGlobalHandlerTy();
  }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { 
    return ELF::EM_X86_64;
  }

  Triple::ArchType getTripleArch() const override {
    return Triple::tpu;
  }

  const char *getName() const override { 
    return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    return true;
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
