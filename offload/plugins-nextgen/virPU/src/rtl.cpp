//===----RTLs/tpu/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for virtual xPU.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include "../dynamic_tpu/pjrt_c_api.h"

#include "Shared/APITypes.h"
#include "Shared/Debug.h"

#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/TargetParser/Triple.h"

using namespace error;

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct VirPUKernelTy;
struct VirPUDeviceTy;
struct VirPUPluginTy;

struct VirPUDeviceImageTy : public DeviceImageTy {
  VirPUDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   std::unique_ptr<MemoryBuffer> Image)
      : DeviceImageTy(ImageId, Device, std::move(Image)) {}
};

struct VirPUKernelTy : public GenericKernelTy {
  VirPUKernelTy(const char *Name) : GenericKernelTy(Name) {}

  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    return Plugin::success();
  }

  Error
  delegatedLaunchImpl(GenericDeviceTy &GenericDevice,
                      std::function<int64_t(void *)> &DelegatedLaunch,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], uint32_t DynBlockMemSize,
                   KernelArgsTy &KernelArgs, KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  /// Return maximum block size for maximum occupancy
  Expected<uint64_t> maxGroupSize(GenericDeviceTy &,
                                  uint64_t DynamicMemSize) const override {
    llvm_unreachable("VirPU Kernel Ty, maxGroupSize!");
  }
};

struct VirPUDeviceTy : public GenericDeviceTy {
  const PJRT_Api* pjrtApi;
  PJRT_Client* pjrtCleint;
  PJRT_Device* pjrtDevice;

  struct PjrtBufferContext {
    PJRT_Buffer* PjrtBuf;
    PJRT_AsyncHostToDeviceTransferManager* TransferManager;
    bool isRealLast; 
  };

  VirPUDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
              PJRT_Api* Api, PJRT_Client* Client, PJRT_Device* Device)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, NVPTXGridValues),
        pjrtApi(Api), pjrtCleint(Client), pjrtDevice(Device) {
    // printf("\nVirPUDeviceTy init success!\n");
  }

  ~VirPUDeviceTy() {}

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

  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    VirPUKernelTy *VirPUKernel = Plugin.allocate<VirPUKernelTy>();
    if (!VirPUKernel)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "failed to allocate memory for VirPU kernel");

    new (VirPUKernel) VirPUKernelTy(Name);

    return *VirPUKernel;
  }

  uint64_t getHardwareParallelism() const override {
    return HardwareParallelism;
  }

  /// We want to set up the RPC server for host services to the GPU if it is
  /// available.
  bool shouldSetupRPCServer() const override { 
    return false; }

  /// Allocate memory on the device or related to the device.
  Expected<void *> allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0) {
      // printf("\n Nothing needs to be allocated!\n");
      return nullptr;
    } 
    // This is the ptr to the target memory
    return malloc(Size);
  }

  /// Deallocate memory on the device or related to the device.
  Error free(void *TgtPtr, TargetAllocTy Kind) override {
    if(!TgtPtr) {
      return Plugin::success();
    }

    typedef void (*DestroyBufFn)(void*, const PJRT_Api*);
    DestroyBufFn destroy_buf = (DestroyBufFn)dlsym(RTLD_DEFAULT, "DestroyPjrtBuffer");
    if (destroy_buf) {
        destroy_buf(TgtPtr, this->pjrtApi);
    } else {
        std::cerr << "Warning: DestroyPjrtBuffer not found in executor.\n";
    }
    std::free(TgtPtr); 
    return Plugin::success();
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override {
    llvm_unreachable("VirPUDeviceTy synchronizeImpl");
    return Plugin::success();
  }

  bool supportVAManagement() const override {
    return true;
  }

  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    return false;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (Size == 0 || TgtPtr == nullptr)
      return Plugin::success();
    // Creating buffer will happen on jit-code-executor side.
    memcpy(TgtPtr, HstPtr, Size);
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (Size == 0) {
      // printf("\nNothing to be retrieved!\n");
      return Plugin::success();
    }

    typedef PJRT_Buffer* (*GetBufFn)(void*);
    GetBufFn get_buf = (GetBufFn)dlsym(RTLD_DEFAULT, "GetPjrtBuffer");

    PJRT_Buffer* PjrtBuf = get_buf(const_cast<void*>(TgtPtr));
    if (PjrtBuf) {
      // Can not use `PJRT_Buffer_CopyRawToHost`, the result would be weird.
      auto args = PJRT_Buffer_ToHostBuffer_Args{
        .struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE,
        .src = PjrtBuf,
        .dst = HstPtr,
        .dst_size = size_t(Size)
      };
      auto* err = this->pjrtApi->PJRT_Buffer_ToHostBuffer(&args);
      assert(!err);

      auto awaitArgs = PJRT_Event_Await_Args{
        .struct_size = PJRT_Event_Await_Args_STRUCT_SIZE,
        .event = args.event
      };
      auto* err2 = this->pjrtApi->PJRT_Event_Await(&awaitArgs);
      assert(!err2);
    }

    return Plugin::success();
   }

  /// Exchange data between two devices directly. We may use peer access if
  /// the CUDA devices and driver allow them.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  Error dataFillImpl(void *TgtPtr, const void *PatternPtr, int64_t PatternSize,
                     int64_t Size,
                     AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("VirPUDeviceTy  dataFillImpl");
  }

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    llvm_unreachable("VirPUDeviceTy initAsyncInfoImpl");
  }

  /// Insert a data fence between previous data operations and the following
  /// operations. This is a no-op for CUDA devices as operations inserted into
  /// a queue are in-order.
  Error dataFence(__tgt_async_info *Async) override {
    llvm_unreachable("VirPUDeviceTy dataFence");
    return Plugin::success();
  }

  interop_spec_t selectInteropPreference(int32_t InteropType,
                                         int32_t NumPrefers,
                                         interop_spec_t *Prefers) override {
    llvm_unreachable("VirPUDeviceTy selectInteropPreference");
    return interop_spec_t{tgt_fr_cuda, {true, 0}, 0};
  }

  Expected<omp_interop_val_t *>
  createInterop(int32_t InteropType, interop_spec_t &InteropSpec) override {
    llvm_unreachable("VirPUDeviceTy createInterop");
  }

  Error releaseInterop(omp_interop_val_t *Interop) override {
    llvm_unreachable("VirPUDeviceTy releaseInterop");
    if (!Interop)
      return Plugin::success();

    if (Interop->async_info)
      delete Interop->async_info;

    delete Interop;
    return Plugin::success();
  }

  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &AsyncInfo) override {
    llvm_unreachable("VirPUDeviceTy enqueueHostCallImpl");
  };

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override {
    return Plugin::success();
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  Expected<bool> isEventCompleteImpl(void *EventPtr,
                                     AsyncInfoWrapperTy &) override {
    return true;
  }

  /// Print information about the device.
  Expected<InfoTreeNode> obtainInfoImpl() override {
    llvm_unreachable("VirPUDeviceTy obtainInfoImpl");
  }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }
  Error setDeviceStackSize(uint64_t Value) override {
    llvm_unreachable("VirPUDeviceTy setDeviceStackSize");
  }
  bool hasDeviceHeapSize() override { 
    return true; 
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }
  Error setDeviceHeapSize(uint64_t Value) override {
    llvm_unreachable("VirPUDeviceTy setDeviceHeapSize");
  }
  Error getDeviceMemorySize(uint64_t &Value) override {
    Value = 80ULL << 30; // 80GB
    return Plugin::success();
  }

  Error getDeviceAttr(uint32_t Kind, uint32_t &Value) {
    llvm_unreachable("VirPUDeviceTy getDeviceAttr");
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override {
    llvm_unreachable("VirPUDeviceTy getComputeUnitKind");
  }

  /// Returns the clock frequency for the given NVPTX device.
  uint64_t getClockFrequency() const override { 
    return 1000000000; }

  Error callGlobalCtorDtorCommon(GenericPluginTy &Plugin, DeviceImageTy &Image,
                                 bool IsCtor) {
    return Plugin::success();
  }

  Expected<DeviceImageTy *>
  loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage,
                 int32_t ImageId) override {
    VirPUDeviceImageTy *VirPUImage = Plugin.allocate<VirPUDeviceImageTy>();
    if (!VirPUImage)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "Failed to allocate memory for VirPU Device Image");
    new (VirPUImage) VirPUDeviceImageTy(ImageId, *this, std::move(TgtImage));
    return VirPUImage;
  }

  Error destroyEventImpl(void *EventPtr) override {
    return Plugin::success();
  }

  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }

  Error syncEventImpl(void *EventPtr) override {
    return Plugin::success();
  }

  Expected<float> getEventElapsedTimeImpl(void *StartEventPtr,
                                          void *EndEventPtr) override {
    return Plugin::success();
  };

  Expected<bool> hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return false; 
  }

  bool useAutoZeroCopyImpl() override {
    return false; 
  }

  Expected<bool> isAccessiblePtrImpl(const void *Ptr, size_t Size) override {
    return false; 
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

Error VirPUKernelTy::delegatedLaunchImpl(
    GenericDeviceTy &GenericDevice,
    std::function<int64_t(void *)> &DelegatedLaunch,
    AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  VirPUDeviceTy &VirPUDevice = static_cast<VirPUDeviceTy &>(GenericDevice);
  Plugin::DelegatedLaunchArgs DLA{
      Plugin::DelegatedLaunchArgs::DeviceTyTy::VirPU, 
      &VirPUDevice,                                    
      nullptr
      // VirPUPlugin->PjrtClient          
  };
  int64_t Res = DelegatedLaunch(&DLA);
  // std::this_thread::sleep_for(std::chrono::seconds(10));
  if (Res)
    return Plugin::error(ErrorCode::UNSUPPORTED, "Error in VirPU delegated launch");
  return Plugin::success();
}

Error VirPUKernelTy::launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], uint32_t DynBlockMemSize,
                   KernelArgsTy &KernelArgs, KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  return Plugin::success();
}

class VirPUGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    return Plugin::success();
  }
};

#define DECLARED_DEVICE_COUNT 1
#define DEVICE_TYPE "cpu"

struct VirPUPluginTy final : public GenericPluginTy {
  // Although claimed to only have one device, VirPUPluginTy actually manage multiple devices, which forms a device mesh.
  // Devices can be logically formed as a multi-dimensional mesh.
  // i.e., a 2x3 mesh means we have 6 devices and logically formed as a 2 row 3 column matrix.
  // TODO: how should I init this DeviceMesh? Currently I can do in initImpl and find all devices and form 1 dimensional.
  llvm::SmallVector<uint8_t> DeviceMesh;
  PJRT_Api* PjrtApi;
  PJRT_Client* PjrtClient = nullptr;

  VirPUPluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  VirPUPluginTy(const VirPUPluginTy &) = delete;
  VirPUPluginTy(VirPUPluginTy &&) = delete;


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

  
  // WARN: this only return a sincle device
  // Get the target device handle
  PJRT_Device *findDevice(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &deviceDescKeyword
  ) {
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
    for (size_t i = 0; i < device_args.num_addressable_devices; i++) {
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
    const char* custom_path = std::getenv("LIBVirPU_PATH");
    void* Handle = nullptr;
    if (custom_path != nullptr) {
      Handle =  dlopen(custom_path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    } else {
      Handle =  dlopen("libtpu.so", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    }
    if (!Handle) {
      printf("VirPU plugin not found, fall back to CPU!\n");
      return 0;
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
    this->PjrtApi = Api;
    
    typedef PJRT_Client* (*GetClientFn)();
    GetClientFn get_client = (GetClientFn)dlsym(RTLD_DEFAULT, "GetExecutorPJRTClient");
    if (get_client) {
      this->PjrtClient = get_client();
    } else {
      PJRT_Plugin_Initialize_Args InitArgs = {};
      InitArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
      auto* InitErr = Api->PJRT_Plugin_Initialize(&InitArgs);
      PJRT_Client_Create_Args args = {
        .struct_size= PJRT_Client_Create_Args_STRUCT_SIZE
      };
      auto* error = Api->PJRT_Client_Create(&args);
      if (error) {
        std::cerr << "Fail to create client!\n";
        std::exit(EXIT_FAILURE);
      }
      this->PjrtClient = args.client;
    }

    SmallVector<PJRT_Device*> Devices = findAllDevices(
      this->PjrtApi, 
      this->PjrtClient, 
      DEVICE_TYPE);



    return DECLARED_DEVICE_COUNT;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    return Plugin::success(); }

  GenericDeviceTy *createDevice(
    GenericPluginTy &Plugin, 
    int32_t DeviceId,
    int32_t NumDevices) override {
    auto* VirPUDevice = findDevice(this->PjrtApi, this->PjrtClient, DEVICE_TYPE);
    return new VirPUDeviceTy(Plugin, DeviceId, NumDevices, this->PjrtApi, this->PjrtClient, VirPUDevice);
  }

  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new VirPUGlobalHandlerTy();
  }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { 
    return ELF::EM_X86_64;
  }

  Triple::ArchType getTripleArch() const override {
    // We actually use x86 here, it does not matter as we will jit execute code rather than compile to VirPU target in LLVM
    return Triple::x86_64;
  }

  const char *getName() const override { 
    return GETNAME(TARGET_NAME); }

  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    return true;
  }


private:
  void mapDevicesToMesh(llvm::SmallVector<PJRT_Device*> Devices) {
    // TODO: how do we get the initial device mesh?
    this->DeviceMesh = {uint8_t(Devices.size())};
    return;
  }

  llvm::SmallVector<PJRT_Device*> findAllDevices(
    const PJRT_Api *api, 
    PJRT_Client *client,
    const std::string &deviceDescKeyword
  ) {
    PJRT_Client_AddressableDevices_Args device_args = {
      .struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
      .client = client,
    };
    llvm::SmallVector<PJRT_Device*> devicesFound;
    auto* err = api->PJRT_Client_AddressableDevices(&device_args);
    if (err || device_args.num_addressable_devices < 1) {
      std::cerr << "no devices found!\n"; 
      return devicesFound;
    }

    for (size_t i = 0; i < device_args.num_addressable_devices; i++) {
      std::string tmp = getDeviceDescription(api, device_args.addressable_devices[i]);
      std::transform(tmp.begin(), tmp.end(), tmp.begin(),[](unsigned char c) { return std::tolower(c); });
      if (tmp.find(deviceDescKeyword) != std::string::npos) {
        devicesFound.push_back(device_args.addressable_devices[i]);
      }
    }
    return devicesFound; 
  }
};

Error VirPUDeviceTy::dataExchangeImpl(const void *SrcPtr,
                                     GenericDeviceTy &DstGenericDevice,
                                     void *DstPtr, int64_t Size,
                                     AsyncInfoWrapperTy &AsyncInfoWrapper) {
  llvm_unreachable("VirPUDeviceTy::dataExchangeImpl");
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
// BUG: this function not created yet
llvm::omp::target::plugin::GenericPluginTy *createPlugin_vpu() {
  return new llvm::omp::target::plugin::VirPUPluginTy();
}
}
