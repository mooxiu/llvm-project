// UNSUPPORTED
// RUN: fir-opt --flatten-target %s | FileCheck %s

// CHECK-LABEL= func.func
func.func private @_QFPrun_benchmark(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.alloca i32
  %1 = fir.alloca i32
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFFrun_benchmarkEi"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.address_of(@_QMomp_lib_kindsECkmp_affinity_mask_kind) : !fir.ref<i32>
  %6:2 = hlfir.declare %5 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsECkmp_affinity_mask_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.address_of(@_QMomp_lib_kindsECkmp_cancel_kind) : !fir.ref<i32>
  %8:2 = hlfir.declare %7 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsECkmp_cancel_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9 = fir.address_of(@_QMomp_libECkmp_cancel_loop) : !fir.ref<i32>
  %10:2 = hlfir.declare %9 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_cancel_loop"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11 = fir.address_of(@_QMomp_libECkmp_cancel_parallel) : !fir.ref<i32>
  %12:2 = hlfir.declare %11 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_cancel_parallel"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %13 = fir.address_of(@_QMomp_libECkmp_cancel_sections) : !fir.ref<i32>
  %14:2 = hlfir.declare %13 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_cancel_sections"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %15 = fir.address_of(@_QMomp_libECkmp_cancel_taskgroup) : !fir.ref<i32>
  %16:2 = hlfir.declare %15 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_cancel_taskgroup"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %17 = fir.address_of(@_QMomp_lib_kindsECkmp_double_kind) : !fir.ref<i32>
  %18:2 = hlfir.declare %17 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsECkmp_double_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %19 = fir.address_of(@_QMomp_libECkmp_lock_hint_adaptive) : !fir.ref<i32>
  %20:2 = hlfir.declare %19 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_lock_hint_adaptive"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %21 = fir.address_of(@_QMomp_libECkmp_lock_hint_hle) : !fir.ref<i32>
  %22:2 = hlfir.declare %21 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_lock_hint_hle"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %23 = fir.address_of(@_QMomp_libECkmp_lock_hint_rtm) : !fir.ref<i32>
  %24:2 = hlfir.declare %23 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_lock_hint_rtm"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %25 = fir.address_of(@_QMomp_lib_kindsECkmp_pointer_kind) : !fir.ref<i32>
  %26:2 = hlfir.declare %25 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsECkmp_pointer_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %27 = fir.address_of(@_QMomp_lib_kindsECkmp_size_t_kind) : !fir.ref<i32>
  %28:2 = hlfir.declare %27 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsECkmp_size_t_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %29 = fir.address_of(@_QMomp_libECkmp_version_build) : !fir.ref<i32>
  %30:2 = hlfir.declare %29 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_version_build"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %31 = fir.address_of(@_QMomp_libECkmp_version_major) : !fir.ref<i32>
  %32:2 = hlfir.declare %31 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_version_major"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %33 = fir.address_of(@_QMomp_libECkmp_version_minor) : !fir.ref<i32>
  %34:2 = hlfir.declare %33 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECkmp_version_minor"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %35 = fir.address_of(@_QMomp_libECllvm_omp_target_device_mem_alloc) : !fir.ref<i64>
  %36:2 = hlfir.declare %35 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_device_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %37 = fir.address_of(@_QMomp_libECllvm_omp_target_device_mem_space) : !fir.ref<i64>
  %38:2 = hlfir.declare %37 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_device_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %39 = fir.address_of(@_QMomp_libECllvm_omp_target_host_mem_alloc) : !fir.ref<i64>
  %40:2 = hlfir.declare %39 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_host_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %41 = fir.address_of(@_QMomp_libECllvm_omp_target_host_mem_space) : !fir.ref<i64>
  %42:2 = hlfir.declare %41 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_host_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %43 = fir.address_of(@_QMomp_libECllvm_omp_target_shared_mem_alloc) : !fir.ref<i64>
  %44:2 = hlfir.declare %43 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_shared_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %45 = fir.address_of(@_QMomp_libECllvm_omp_target_shared_mem_space) : !fir.ref<i64>
  %46:2 = hlfir.declare %45 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECllvm_omp_target_shared_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %47:2 = hlfir.declare %arg2 dummy_scope %2 arg 3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %48 = fir.address_of(@_QMomp_lib_kindsEComp_allocator_handle_kind) : !fir.ref<i32>
  %49:2 = hlfir.declare %48 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_allocator_handle_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %50 = fir.address_of(@_QMomp_lib_kindsEComp_alloctrait_key_kind) : !fir.ref<i32>
  %51:2 = hlfir.declare %50 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_alloctrait_key_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %52 = fir.address_of(@_QMomp_lib_kindsEComp_alloctrait_val_kind) : !fir.ref<i32>
  %53:2 = hlfir.declare %52 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_alloctrait_val_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %54 = fir.address_of(@_QMomp_libEComp_atk_access) : !fir.ref<i32>
  %55:2 = hlfir.declare %54 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_access"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %56 = fir.address_of(@_QMomp_libEComp_atk_alignment) : !fir.ref<i32>
  %57:2 = hlfir.declare %56 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_alignment"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %58 = fir.address_of(@_QMomp_libEComp_atk_atomic_scope) : !fir.ref<i32>
  %59:2 = hlfir.declare %58 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_atomic_scope"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %60 = fir.address_of(@_QMomp_libEComp_atk_device_access) : !fir.ref<i32>
  %61:2 = hlfir.declare %60 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_device_access"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %62 = fir.address_of(@_QMomp_libEComp_atk_fallback) : !fir.ref<i32>
  %63:2 = hlfir.declare %62 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_fallback"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %64 = fir.address_of(@_QMomp_libEComp_atk_fb_data) : !fir.ref<i32>
  %65:2 = hlfir.declare %64 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_fb_data"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %66 = fir.address_of(@_QMomp_libEComp_atk_part_size) : !fir.ref<i32>
  %67:2 = hlfir.declare %66 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_part_size"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %68 = fir.address_of(@_QMomp_libEComp_atk_partition) : !fir.ref<i32>
  %69:2 = hlfir.declare %68 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_partition"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %70 = fir.address_of(@_QMomp_libEComp_atk_pin_device) : !fir.ref<i32>
  %71:2 = hlfir.declare %70 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_pin_device"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %72 = fir.address_of(@_QMomp_libEComp_atk_pinned) : !fir.ref<i32>
  %73:2 = hlfir.declare %72 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_pinned"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %74 = fir.address_of(@_QMomp_libEComp_atk_pool_size) : !fir.ref<i32>
  %75:2 = hlfir.declare %74 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_pool_size"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %76 = fir.address_of(@_QMomp_libEComp_atk_preferred_device) : !fir.ref<i32>
  %77:2 = hlfir.declare %76 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_preferred_device"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %78 = fir.address_of(@_QMomp_libEComp_atk_sync_hint) : !fir.ref<i32>
  %79:2 = hlfir.declare %78 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_sync_hint"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %80 = fir.address_of(@_QMomp_libEComp_atk_target_access) : !fir.ref<i32>
  %81:2 = hlfir.declare %80 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atk_target_access"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %82 = fir.address_of(@_QMomp_libEComp_atv_abort_fb) : !fir.ref<i64>
  %83:2 = hlfir.declare %82 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_abort_fb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %84 = fir.address_of(@_QMomp_libEComp_atv_all) : !fir.ref<i64>
  %85:2 = hlfir.declare %84 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_all"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %86 = fir.address_of(@_QMomp_libEComp_atv_allocator_fb) : !fir.ref<i64>
  %87:2 = hlfir.declare %86 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_allocator_fb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %88 = fir.address_of(@_QMomp_libEComp_atv_blocked) : !fir.ref<i64>
  %89:2 = hlfir.declare %88 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_blocked"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %90 = fir.address_of(@_QMomp_libEComp_atv_cgroup) : !fir.ref<i64>
  %91:2 = hlfir.declare %90 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_cgroup"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %92 = fir.address_of(@_QMomp_libEComp_atv_contended) : !fir.ref<i64>
  %93:2 = hlfir.declare %92 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_contended"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %94 = fir.address_of(@_QMomp_libEComp_atv_default) : !fir.ref<i64>
  %95:2 = hlfir.declare %94 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_default"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %96 = fir.address_of(@_QMomp_libEComp_atv_default_mem_fb) : !fir.ref<i64>
  %97:2 = hlfir.declare %96 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_default_mem_fb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %98 = fir.address_of(@_QMomp_libEComp_atv_device) : !fir.ref<i64>
  %99:2 = hlfir.declare %98 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_device"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %100 = fir.address_of(@_QMomp_libEComp_atv_environment) : !fir.ref<i64>
  %101:2 = hlfir.declare %100 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_environment"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %102 = fir.address_of(@_QMomp_libEComp_atv_false) : !fir.ref<i64>
  %103:2 = hlfir.declare %102 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_false"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %104 = fir.address_of(@_QMomp_libEComp_atv_interleaved) : !fir.ref<i64>
  %105:2 = hlfir.declare %104 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_interleaved"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %106 = fir.address_of(@_QMomp_libEComp_atv_memspace) : !fir.ref<i64>
  %107:2 = hlfir.declare %106 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_memspace"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %108 = fir.address_of(@_QMomp_libEComp_atv_multiple) : !fir.ref<i64>
  %109:2 = hlfir.declare %108 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_multiple"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %110 = fir.address_of(@_QMomp_libEComp_atv_nearest) : !fir.ref<i64>
  %111:2 = hlfir.declare %110 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_nearest"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %112 = fir.address_of(@_QMomp_libEComp_atv_null_fb) : !fir.ref<i64>
  %113:2 = hlfir.declare %112 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_null_fb"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %114 = fir.address_of(@_QMomp_libEComp_atv_private) : !fir.ref<i64>
  %115:2 = hlfir.declare %114 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_private"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %116 = fir.address_of(@_QMomp_libEComp_atv_pteam) : !fir.ref<i64>
  %117:2 = hlfir.declare %116 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_pteam"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %118 = fir.address_of(@_QMomp_libEComp_atv_sequential) : !fir.ref<i64>
  %119:2 = hlfir.declare %118 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_sequential"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %120 = fir.address_of(@_QMomp_libEComp_atv_serialized) : !fir.ref<i64>
  %121:2 = hlfir.declare %120 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_serialized"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %122 = fir.address_of(@_QMomp_libEComp_atv_single) : !fir.ref<i64>
  %123:2 = hlfir.declare %122 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_single"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %124 = fir.address_of(@_QMomp_libEComp_atv_thread) : !fir.ref<i64>
  %125:2 = hlfir.declare %124 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_thread"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %126 = fir.address_of(@_QMomp_libEComp_atv_true) : !fir.ref<i64>
  %127:2 = hlfir.declare %126 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_true"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %128 = fir.address_of(@_QMomp_libEComp_atv_uncontended) : !fir.ref<i64>
  %129:2 = hlfir.declare %128 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_atv_uncontended"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %130 = fir.address_of(@_QMomp_libEComp_cgroup_mem_alloc) : !fir.ref<i64>
  %131:2 = hlfir.declare %130 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_cgroup_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %132 = fir.address_of(@_QMomp_libEComp_const_mem_alloc) : !fir.ref<i64>
  %133:2 = hlfir.declare %132 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_const_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %134 = fir.address_of(@_QMomp_libEComp_const_mem_space) : !fir.ref<i64>
  %135:2 = hlfir.declare %134 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_const_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %136 = fir.address_of(@_QMomp_libEComp_control_tool_end) : !fir.ref<i32>
  %137:2 = hlfir.declare %136 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_end"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %138 = fir.address_of(@_QMomp_libEComp_control_tool_flush) : !fir.ref<i32>
  %139:2 = hlfir.declare %138 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_flush"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %140 = fir.address_of(@_QMomp_libEComp_control_tool_ignored) : !fir.ref<i32>
  %141:2 = hlfir.declare %140 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_ignored"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %142 = fir.address_of(@_QMomp_lib_kindsEComp_control_tool_kind) : !fir.ref<i32>
  %143:2 = hlfir.declare %142 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_control_tool_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %144 = fir.address_of(@_QMomp_libEComp_control_tool_nocallback) : !fir.ref<i32>
  %145:2 = hlfir.declare %144 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_nocallback"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %146 = fir.address_of(@_QMomp_libEComp_control_tool_notool) : !fir.ref<i32>
  %147:2 = hlfir.declare %146 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_notool"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %148 = fir.address_of(@_QMomp_libEComp_control_tool_pause) : !fir.ref<i32>
  %149:2 = hlfir.declare %148 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_pause"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %150 = fir.address_of(@_QMomp_lib_kindsEComp_control_tool_result_kind) : !fir.ref<i32>
  %151:2 = hlfir.declare %150 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_control_tool_result_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %152 = fir.address_of(@_QMomp_libEComp_control_tool_start) : !fir.ref<i32>
  %153:2 = hlfir.declare %152 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_start"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %154 = fir.address_of(@_QMomp_libEComp_control_tool_success) : !fir.ref<i32>
  %155:2 = hlfir.declare %154 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_control_tool_success"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %156 = fir.address_of(@_QMomp_libEComp_default_mem_alloc) : !fir.ref<i64>
  %157:2 = hlfir.declare %156 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_default_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %158 = fir.address_of(@_QMomp_libEComp_default_mem_space) : !fir.ref<i64>
  %159:2 = hlfir.declare %158 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_default_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %160 = fir.address_of(@_QMomp_lib_kindsEComp_depend_kind) : !fir.ref<i32>
  %161:2 = hlfir.declare %160 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_depend_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %162 = fir.address_of(@_QMomp_lib_kindsEComp_event_handle_kind) : !fir.ref<i32>
  %163:2 = hlfir.declare %162 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_event_handle_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %164 = fir.address_of(@_QMomp_libEComp_high_bw_mem_alloc) : !fir.ref<i64>
  %165:2 = hlfir.declare %164 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_high_bw_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %166 = fir.address_of(@_QMomp_libEComp_high_bw_mem_space) : !fir.ref<i64>
  %167:2 = hlfir.declare %166 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_high_bw_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %168 = fir.address_of(@_QMomp_libEComp_ifr_cuda) : !fir.ref<i32>
  %169:2 = hlfir.declare %168 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_cuda"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %170 = fir.address_of(@_QMomp_libEComp_ifr_cuda_driver) : !fir.ref<i32>
  %171:2 = hlfir.declare %170 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_cuda_driver"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %172 = fir.address_of(@_QMomp_libEComp_ifr_hip) : !fir.ref<i32>
  %173:2 = hlfir.declare %172 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_hip"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %174 = fir.address_of(@_QMomp_libEComp_ifr_last) : !fir.ref<i32>
  %175:2 = hlfir.declare %174 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_last"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %176 = fir.address_of(@_QMomp_libEComp_ifr_level_zero) : !fir.ref<i32>
  %177:2 = hlfir.declare %176 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_level_zero"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %178 = fir.address_of(@_QMomp_libEComp_ifr_opencl) : !fir.ref<i32>
  %179:2 = hlfir.declare %178 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_opencl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %180 = fir.address_of(@_QMomp_libEComp_ifr_sycl) : !fir.ref<i32>
  %181:2 = hlfir.declare %180 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_ifr_sycl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %182 = fir.address_of(@_QMomp_lib_kindsEComp_integer_kind) : !fir.ref<i32>
  %183:2 = hlfir.declare %182 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_integer_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %184 = fir.address_of(@_QMomp_lib_kindsEComp_interop_fr_kind) : !fir.ref<i32>
  %185:2 = hlfir.declare %184 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_interop_fr_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %186 = fir.address_of(@_QMomp_lib_kindsEComp_interop_kind) : !fir.ref<i32>
  %187:2 = hlfir.declare %186 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_interop_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %188 = fir.address_of(@_QMomp_libEComp_interop_none) : !fir.ref<i64>
  %189:2 = hlfir.declare %188 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_interop_none"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %190 = fir.address_of(@_QMomp_libEComp_large_cap_mem_alloc) : !fir.ref<i64>
  %191:2 = hlfir.declare %190 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_large_cap_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %192 = fir.address_of(@_QMomp_libEComp_large_cap_mem_space) : !fir.ref<i64>
  %193:2 = hlfir.declare %192 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_large_cap_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %194 = fir.address_of(@_QMomp_libEComp_lock_hint_contended) : !fir.ref<i32>
  %195:2 = hlfir.declare %194 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_lock_hint_contended"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %196 = fir.address_of(@_QMomp_lib_kindsEComp_lock_hint_kind) : !fir.ref<i32>
  %197:2 = hlfir.declare %196 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_lock_hint_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %198 = fir.address_of(@_QMomp_libEComp_lock_hint_none) : !fir.ref<i32>
  %199:2 = hlfir.declare %198 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_lock_hint_none"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %200 = fir.address_of(@_QMomp_libEComp_lock_hint_nonspeculative) : !fir.ref<i32>
  %201:2 = hlfir.declare %200 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_lock_hint_nonspeculative"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %202 = fir.address_of(@_QMomp_libEComp_lock_hint_speculative) : !fir.ref<i32>
  %203:2 = hlfir.declare %202 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_lock_hint_speculative"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %204 = fir.address_of(@_QMomp_libEComp_lock_hint_uncontended) : !fir.ref<i32>
  %205:2 = hlfir.declare %204 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_lock_hint_uncontended"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %206 = fir.address_of(@_QMomp_lib_kindsEComp_lock_kind) : !fir.ref<i32>
  %207:2 = hlfir.declare %206 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_lock_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %208 = fir.address_of(@_QMomp_lib_kindsEComp_logical_kind) : !fir.ref<i32>
  %209:2 = hlfir.declare %208 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_logical_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %210 = fir.address_of(@_QMomp_libEComp_low_lat_mem_alloc) : !fir.ref<i64>
  %211:2 = hlfir.declare %210 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_low_lat_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %212 = fir.address_of(@_QMomp_libEComp_low_lat_mem_space) : !fir.ref<i64>
  %213:2 = hlfir.declare %212 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_low_lat_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %214 = fir.address_of(@_QMomp_lib_kindsEComp_memspace_handle_kind) : !fir.ref<i32>
  %215:2 = hlfir.declare %214 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_memspace_handle_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %216 = fir.address_of(@_QMomp_lib_kindsEComp_nest_lock_kind) : !fir.ref<i32>
  %217:2 = hlfir.declare %216 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_nest_lock_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %218 = fir.address_of(@_QMomp_libEComp_null_allocator) : !fir.ref<i64>
  %219:2 = hlfir.declare %218 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_null_allocator"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %220 = fir.address_of(@_QMomp_libEComp_null_mem_space) : !fir.ref<i64>
  %221:2 = hlfir.declare %220 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_null_mem_space"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %222 = fir.address_of(@_QMomp_libEComp_pause_hard) : !fir.ref<i32>
  %223:2 = hlfir.declare %222 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_pause_hard"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %224 = fir.address_of(@_QMomp_lib_kindsEComp_pause_resource_kind) : !fir.ref<i32>
  %225:2 = hlfir.declare %224 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_pause_resource_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %226 = fir.address_of(@_QMomp_libEComp_pause_resume) : !fir.ref<i32>
  %227:2 = hlfir.declare %226 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_pause_resume"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %228 = fir.address_of(@_QMomp_libEComp_pause_soft) : !fir.ref<i32>
  %229:2 = hlfir.declare %228 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_pause_soft"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %230 = fir.address_of(@_QMomp_libEComp_pause_stop_tool) : !fir.ref<i32>
  %231:2 = hlfir.declare %230 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_pause_stop_tool"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %232 = fir.address_of(@_QMomp_libEComp_proc_bind_close) : !fir.ref<i32>
  %233:2 = hlfir.declare %232 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_proc_bind_close"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %234 = fir.address_of(@_QMomp_libEComp_proc_bind_false) : !fir.ref<i32>
  %235:2 = hlfir.declare %234 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_proc_bind_false"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %236 = fir.address_of(@_QMomp_lib_kindsEComp_proc_bind_kind) : !fir.ref<i32>
  %237:2 = hlfir.declare %236 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_proc_bind_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %238 = fir.address_of(@_QMomp_libEComp_proc_bind_master) : !fir.ref<i32>
  %239:2 = hlfir.declare %238 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_proc_bind_master"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %240 = fir.address_of(@_QMomp_libEComp_proc_bind_spread) : !fir.ref<i32>
  %241:2 = hlfir.declare %240 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_proc_bind_spread"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %242 = fir.address_of(@_QMomp_libEComp_proc_bind_true) : !fir.ref<i32>
  %243:2 = hlfir.declare %242 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_proc_bind_true"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %244 = fir.address_of(@_QMomp_libEComp_pteam_mem_alloc) : !fir.ref<i64>
  %245:2 = hlfir.declare %244 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_pteam_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %246 = fir.address_of(@_QMomp_lib_kindsEComp_real_kind) : !fir.ref<i32>
  %247:2 = hlfir.declare %246 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_real_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %248 = fir.address_of(@_QMomp_libEComp_sched_auto) : !fir.ref<i32>
  %249:2 = hlfir.declare %248 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sched_auto"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %250 = fir.address_of(@_QMomp_libEComp_sched_dynamic) : !fir.ref<i32>
  %251:2 = hlfir.declare %250 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sched_dynamic"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %252 = fir.address_of(@_QMomp_libEComp_sched_guided) : !fir.ref<i32>
  %253:2 = hlfir.declare %252 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sched_guided"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %254 = fir.address_of(@_QMomp_lib_kindsEComp_sched_kind) : !fir.ref<i32>
  %255:2 = hlfir.declare %254 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_sched_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %256 = fir.address_of(@_QMomp_libEComp_sched_monotonic) : !fir.ref<i32>
  %257:2 = hlfir.declare %256 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sched_monotonic"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %258 = fir.address_of(@_QMomp_libEComp_sched_static) : !fir.ref<i32>
  %259:2 = hlfir.declare %258 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sched_static"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %260 = fir.address_of(@_QMomp_libEComp_sync_hint_contended) : !fir.ref<i32>
  %261:2 = hlfir.declare %260 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sync_hint_contended"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %262 = fir.address_of(@_QMomp_lib_kindsEComp_sync_hint_kind) : !fir.ref<i32>
  %263:2 = hlfir.declare %262 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_lib_kindsEComp_sync_hint_kind"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %264 = fir.address_of(@_QMomp_libEComp_sync_hint_none) : !fir.ref<i32>
  %265:2 = hlfir.declare %264 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sync_hint_none"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %266 = fir.address_of(@_QMomp_libEComp_sync_hint_nonspeculative) : !fir.ref<i32>
  %267:2 = hlfir.declare %266 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sync_hint_nonspeculative"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %268 = fir.address_of(@_QMomp_libEComp_sync_hint_speculative) : !fir.ref<i32>
  %269:2 = hlfir.declare %268 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sync_hint_speculative"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %270 = fir.address_of(@_QMomp_libEComp_sync_hint_uncontended) : !fir.ref<i32>
  %271:2 = hlfir.declare %270 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_sync_hint_uncontended"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %272 = fir.address_of(@_QMomp_libEComp_thread_mem_alloc) : !fir.ref<i64>
  %273:2 = hlfir.declare %272 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libEComp_thread_mem_alloc"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %274 = fir.address_of(@_QMomp_libECopenmp_version) : !fir.ref<i32>
  %275:2 = hlfir.declare %274 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMomp_libECopenmp_version"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %276 = fir.load %47#0 : !fir.ref<i32>
  fir.store %276 to %1 : !fir.ref<i32>
  %277 = fir.convert %276 : (i32) -> i64
  %278 = fir.convert %277 : (i64) -> index
  %c0 = arith.constant 0 : index
  %279 = arith.cmpi sgt, %278, %c0 : index
  %280 = arith.select %279, %278, %c0 : index
  %281 = fir.shape %280 : (index) -> !fir.shape<1>
  %282:2 = hlfir.declare %arg0(%281) dummy_scope %2 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %283 = fir.load %47#0 : !fir.ref<i32>
  fir.store %283 to %0 : !fir.ref<i32>
  %284 = fir.convert %283 : (i32) -> i64
  %285 = fir.convert %284 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %286 = arith.cmpi sgt, %285, %c0_0 : index
  %287 = arith.select %286, %285, %c0_0 : index
  %288 = fir.shape %287 : (index) -> !fir.shape<1>
  %289:2 = hlfir.declare %arg1(%288) dummy_scope %2 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
  %c1_i32 = arith.constant 1 : i32
  %290 = fir.load %47#0 : !fir.ref<i32>
  %c1_i32_1 = arith.constant 1 : i32
  %291 = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
  %292 = omp.map.info var_ptr(%47#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "n"}
  %c1 = arith.constant 1 : index
  %c0_2 = arith.constant 0 : index
  %293 = arith.subi %287, %c1 : index
  %294 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%293 : index) extent(%287 : index) stride(%c1 : index) start_idx(%c1 : index)
  %295 = omp.map.info var_ptr(%289#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%294) -> !fir.ref<!fir.array<?xf64>> {name = "y"}
  %c1_3 = arith.constant 1 : index
  %c0_4 = arith.constant 0 : index
  %296 = arith.subi %280, %c1_3 : index
  %297 = omp.map.bounds lower_bound(%c0_4 : index) upper_bound(%296 : index) extent(%280 : index) stride(%c1_3 : index) start_idx(%c1_3 : index)
  %298 = omp.map.info var_ptr(%282#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%297) -> !fir.ref<!fir.array<?xf64>> {name = "x"}
  %299 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
  %300 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
  // CHECK: omp.target
  // CHECK-SAME: map_entries
  // CHECK-NOT: host_eval 
  // CHECK-NOT: omp.teams
  // CHECK-NOT: omp.parallel
  // CHECK-NOT: omp.distribute
  // CHECK-NOT: omp.wsloop
  // CHECK-NOT: omp.loop_nest
  // CHECK-NOT: private
  // CHECK: fir.do_loop
  omp.target host_eval(%c1_i32 -> %arg3, %290 -> %arg4, %c1_i32_1 -> %arg5 : i32, i32, i32) map_entries(%291 -> %arg6, %292 -> %arg7, %295 -> %arg8, %298 -> %arg9, %299 -> %arg10, %300 -> %arg11 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>, !fir.ref<i32>, !fir.ref<i32>) {
    %301 = fir.load %arg11 : !fir.ref<i32>
    %302 = fir.load %arg10 : !fir.ref<i32>
    %303 = fir.convert %302 : (i32) -> i64
    %304 = fir.convert %301 : (i32) -> i64
    %c0_5 = arith.constant 0 : index
    %305 = fir.convert %304 : (i64) -> index
    %306 = arith.cmpi sgt, %305, %c0_5 : index
    %c0_6 = arith.constant 0 : index
    %307 = fir.convert %303 : (i64) -> index
    %308 = arith.cmpi sgt, %307, %c0_6 : index
    %309 = arith.select %308, %307, %c0_6 : index
    %310 = arith.select %306, %305, %c0_5 : index
    %311:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %312:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %313 = fir.shape %310 : (index) -> !fir.shape<1>
    %314:2 = hlfir.declare %arg8(%313) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %315 = fir.shape %309 : (index) -> !fir.shape<1>
    %316:2 = hlfir.declare %arg9(%315) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    omp.teams {
      omp.parallel private(@_QFFrun_benchmarkEi_private_i32 %311#0 -> %arg12 : !fir.ref<i32>) {
        %317:2 = hlfir.declare %arg12 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
        omp.distribute {
          omp.wsloop {
            omp.loop_nest (%arg13) : i32 = (%arg3) to (%arg4) inclusive step (%arg5) {
              hlfir.assign %arg13 to %317#0 : i32, !fir.ref<i32>
              %318 = fir.load %317#0 : !fir.ref<i32>
              %319 = fir.convert %318 : (i32) -> i64
              %320 = hlfir.designate %316#0 (%319)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
              %321 = fir.load %320 : !fir.ref<f64>
              %322 = fir.load %317#0 : !fir.ref<i32>
              %323 = fir.convert %322 : (i32) -> i64
              %324 = hlfir.designate %314#0 (%323)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
              hlfir.assign %321 to %324 : f64, !fir.ref<f64>
              omp.yield
            }
          } {omp.composite}
        } {omp.composite}
        omp.terminator
      } {omp.composite}
      omp.terminator
    }
    omp.terminator
  }
  return
}
