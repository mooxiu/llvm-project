// RUN: split-file %s %t
// RUN: fir-opt %t/trivial-assign.mlir --flatten-target | FileCheck %s --check-prefix=TRIVIAL-ASSIGN
// RUN: fir-opt %t/single-loop-assign.mlir --flatten-target | FileCheck %s --check-prefix=SL-ASSIGN
// RUN: fir-opt %t/nested-loop-assign.mlir --flatten-target | FileCheck %s --check-prefix=DL-ASSIGN
// RUN: fir-opt %t/single-loop-reduction.mlir --flatten-target | FileCheck %s --check-prefix=SL-REDUCTION


//--- trivial-assign.mlir
// TRIVIAL-ASSIGN-LABEL: func.func private @_QFPrun_benchmark
module {
  func.func private @_QFPrun_benchmark(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
    %2:2 = hlfir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
    %3 = omp.map.info var_ptr(%2#1 : !fir.ref<f64>, f64) map_clauses(implicit) capture(ByCopy) -> !fir.ref<f64> {name = "y"}
    %4 = omp.map.info var_ptr(%1#1 : !fir.ref<f64>, f64) map_clauses(implicit) capture(ByCopy) -> !fir.ref<f64> {name = "x"}
    // TRIVIAL-ASSIGN: omp.target
    // TRIVIAL-ASSIGN-SAME: map_entries
    // TRIVIAL-ASSIGN-SAME: %3 -> %arg2
    // TRIVIAL-ASSIGN-SAME: %4 -> %arg3
    omp.target map_entries(%3 -> %arg2, %4 -> %arg3 : !fir.ref<f64>, !fir.ref<f64>) {
      %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
      %6:2 = hlfir.declare %arg3 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
      %7 = fir.load %6#0 : !fir.ref<f64>
      hlfir.assign %7 to %5#0 : f64, !fir.ref<f64>
      omp.terminator
    }
    return
  }
}


//--- single-loop-assign.mlir
// SL-ASSIGN-LABEL: func.func private @_QFPrun_benchmark
module {
  omp.private {type = private} @_QFFrun_benchmarkEi_private_i32 : i32
  func.func private @_QFPrun_benchmark(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.alloca i32
    %1 = fir.alloca i32
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFFrun_benchmarkEi"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg2 dummy_scope %2 arg 3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %6 = fir.load %5#0 : !fir.ref<i32>
    fir.store %6 to %1 : !fir.ref<i32>
    %7 = fir.convert %6 : (i32) -> i64
    %8 = fir.convert %7 : (i64) -> index
    %c0 = arith.constant 0 : index
    %9 = arith.cmpi sgt, %8, %c0 : index
    %10 = arith.select %9, %8, %c0 : index
    %11 = fir.shape %10 : (index) -> !fir.shape<1>
    %12:2 = hlfir.declare %arg0(%11) dummy_scope %2 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %13 = fir.load %5#0 : !fir.ref<i32>
    fir.store %13 to %0 : !fir.ref<i32>
    %14 = fir.convert %13 : (i32) -> i64
    %15 = fir.convert %14 : (i64) -> index
    %c0_0 = arith.constant 0 : index
    %16 = arith.cmpi sgt, %15, %c0_0 : index
    %17 = arith.select %16, %15, %c0_0 : index
    %18 = fir.shape %17 : (index) -> !fir.shape<1>
    %19:2 = hlfir.declare %arg1(%18) dummy_scope %2 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %c1_i32 = arith.constant 1 : i32
    %20 = fir.load %5#0 : !fir.ref<i32>
    %c1_i32_1 = arith.constant 1 : i32
    %21 = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
    %22 = omp.map.info var_ptr(%5#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "n"}
    %c1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %23 = arith.subi %17, %c1 : index
    %24 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%23 : index) extent(%17 : index) stride(%c1 : index) start_idx(%c1 : index)
    %25 = omp.map.info var_ptr(%19#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%24) -> !fir.ref<!fir.array<?xf64>> {name = "y"}
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %26 = arith.subi %10, %c1_3 : index
    %27 = omp.map.bounds lower_bound(%c0_4 : index) upper_bound(%26 : index) extent(%10 : index) stride(%c1_3 : index) start_idx(%c1_3 : index)
    %28 = omp.map.info var_ptr(%12#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%27) -> !fir.ref<!fir.array<?xf64>> {name = "x"}
    %29 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %30 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
  // SL-ASSIGN: omp.target
  // SL-ASSIGN-NOT: host_eval
  // SL-ASSIGN-SAME: map_entries
  // SL-ASSIGN-NOT: private(
  // SL-ASSIGN-NOT: omp.teams
  // SL-ASSIGN-NOT: omp.parallel
  // SL-ASSIGN-NOT: omp.distribute
  // SL-ASSIGN-NOT: omp.wsloop
  // SL-ASSIGN-NOT: omp.loop_nest
    omp.target host_eval(%c1_i32 -> %arg3, %20 -> %arg4, %c1_i32_1 -> %arg5 : i32, i32, i32) map_entries(%21 -> %arg6, %22 -> %arg7, %25 -> %arg8, %28 -> %arg9, %29 -> %arg10, %30 -> %arg11 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>, !fir.ref<i32>, !fir.ref<i32>) {
      %31 = fir.load %arg11 : !fir.ref<i32>
      %32 = fir.load %arg10 : !fir.ref<i32>
      %33 = fir.convert %32 : (i32) -> i64
      %34 = fir.convert %31 : (i32) -> i64
      %c0_5 = arith.constant 0 : index
      %35 = fir.convert %34 : (i64) -> index
      %36 = arith.cmpi sgt, %35, %c0_5 : index
      %c0_6 = arith.constant 0 : index
      %37 = fir.convert %33 : (i64) -> index
      %38 = arith.cmpi sgt, %37, %c0_6 : index
      %39 = arith.select %38, %37, %c0_6 : index
      %40 = arith.select %36, %35, %c0_5 : index
      %41:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %42:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %43 = fir.shape %40 : (index) -> !fir.shape<1>
      %44:2 = hlfir.declare %arg8(%43) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      %45 = fir.shape %39 : (index) -> !fir.shape<1>
      %46:2 = hlfir.declare %arg9(%45) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      omp.teams {
        omp.parallel private(@_QFFrun_benchmarkEi_private_i32 %41#0 -> %arg12 : !fir.ref<i32>) {
          %47:2 = hlfir.declare %arg12 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg13) : i32 = (%arg3) to (%arg4) inclusive step (%arg5) {
                hlfir.assign %arg13 to %47#0 : i32, !fir.ref<i32>
                %48 = fir.load %47#0 : !fir.ref<i32>
                %49 = fir.convert %48 : (i32) -> i64
                %50 = hlfir.designate %46#0 (%49)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
                %51 = fir.load %50 : !fir.ref<f64>
                %52 = fir.load %47#0 : !fir.ref<i32>
                %53 = fir.convert %52 : (i32) -> i64
                %54 = hlfir.designate %44#0 (%53)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
                hlfir.assign %51 to %54 : f64, !fir.ref<f64>
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    return
  }
}


//--- nested-loop-assign.mlir
// DL-ASSIGN-LABEL: func.func private
module{
  omp.private {type = private} @_QFFrun_benchmarkEj_private_i32 : i32
  omp.private {type = private} @_QFFrun_benchmarkEi_private_i32 : i32
  func.func private @_QFPrun_benchmark(%arg0: !fir.ref<!fir.array<?x?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?x?xf64>> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.alloca i32
    %1 = fir.alloca i32
    %2 = fir.alloca i32
    %3 = fir.alloca i32
    %4 = fir.dummy_scope : !fir.dscope
    %5 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFFrun_benchmarkEi"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %7 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFFrun_benchmarkEj"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9:2 = hlfir.declare %arg2 dummy_scope %4 arg 3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %10 = fir.load %9#0 : !fir.ref<i32>
    fir.store %10 to %2 : !fir.ref<i32>
    %11 = fir.convert %10 : (i32) -> i64
    %12 = fir.convert %11 : (i64) -> index
    %c0 = arith.constant 0 : index
    %13 = arith.cmpi sgt, %12, %c0 : index
    %14 = arith.select %13, %12, %c0 : index
    %15 = fir.load %9#0 : !fir.ref<i32>
    fir.store %15 to %3 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.convert %16 : (i64) -> index
    %c0_0 = arith.constant 0 : index
    %18 = arith.cmpi sgt, %17, %c0_0 : index
    %19 = arith.select %18, %17, %c0_0 : index
    %20 = fir.shape %14, %19 : (index, index) -> !fir.shape<2>
    %21:2 = hlfir.declare %arg0(%20) dummy_scope %4 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %22 = fir.load %9#0 : !fir.ref<i32>
    fir.store %22 to %0 : !fir.ref<i32>
    %23 = fir.convert %22 : (i32) -> i64
    %24 = fir.convert %23 : (i64) -> index
    %c0_1 = arith.constant 0 : index
    %25 = arith.cmpi sgt, %24, %c0_1 : index
    %26 = arith.select %25, %24, %c0_1 : index
    %27 = fir.load %9#0 : !fir.ref<i32>
    fir.store %27 to %1 : !fir.ref<i32>
    %28 = fir.convert %27 : (i32) -> i64
    %29 = fir.convert %28 : (i64) -> index
    %c0_2 = arith.constant 0 : index
    %30 = arith.cmpi sgt, %29, %c0_2 : index
    %31 = arith.select %30, %29, %c0_2 : index
    %32 = fir.shape %26, %31 : (index, index) -> !fir.shape<2>
    %33:2 = hlfir.declare %arg1(%32) dummy_scope %4 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
    %c1_i32 = arith.constant 1 : i32
    %34 = fir.load %9#0 : !fir.ref<i32>
    %c1_i32_3 = arith.constant 1 : i32
    %35 = omp.map.info var_ptr(%6#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
    %36 = omp.map.info var_ptr(%9#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "n"}
    %37 = omp.map.info var_ptr(%8#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "j"}
    %c1 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %38 = arith.subi %26, %c1 : index
    %39 = omp.map.bounds lower_bound(%c0_4 : index) upper_bound(%38 : index) extent(%26 : index) stride(%c1 : index) start_idx(%c1 : index)
    %c0_5 = arith.constant 0 : index
    %40 = arith.subi %31, %c1 : index
    %41 = omp.map.bounds lower_bound(%c0_5 : index) upper_bound(%40 : index) extent(%31 : index) stride(%c1 : index) start_idx(%c1 : index)
    %42 = omp.map.info var_ptr(%33#1 : !fir.ref<!fir.array<?x?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%39, %41) -> !fir.ref<!fir.array<?x?xf64>> {name = "y"}
    %c1_6 = arith.constant 1 : index
    %c0_7 = arith.constant 0 : index
    %43 = arith.subi %14, %c1_6 : index
    %44 = omp.map.bounds lower_bound(%c0_7 : index) upper_bound(%43 : index) extent(%14 : index) stride(%c1_6 : index) start_idx(%c1_6 : index)
    %c0_8 = arith.constant 0 : index
    %45 = arith.subi %19, %c1_6 : index
    %46 = omp.map.bounds lower_bound(%c0_8 : index) upper_bound(%45 : index) extent(%19 : index) stride(%c1_6 : index) start_idx(%c1_6 : index)
    %47 = omp.map.info var_ptr(%21#1 : !fir.ref<!fir.array<?x?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%44, %46) -> !fir.ref<!fir.array<?x?xf64>> {name = "x"}
    %48 = omp.map.info var_ptr(%3 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %49 = omp.map.info var_ptr(%2 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %50 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %51 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    // DL-ASSIGN-LABEL: omp.target
    // DL-ASSIGN-NOT: host_eval
    // DL-ASSIGN-SAME: map_entries
    omp.target host_eval(%c1_i32 -> %arg3, %34 -> %arg4, %c1_i32_3 -> %arg5 : i32, i32, i32) map_entries(%35 -> %arg6, %36 -> %arg7, %37 -> %arg8, %42 -> %arg9, %47 -> %arg10, %48 -> %arg11, %49 -> %arg12, %50 -> %arg13, %51 -> %arg14 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
      %52 = fir.load %arg14 : !fir.ref<i32>
      %53 = fir.load %arg13 : !fir.ref<i32>
      %54 = fir.load %arg12 : !fir.ref<i32>
      %55 = fir.load %arg11 : !fir.ref<i32>
      %56 = fir.convert %55 : (i32) -> i64
      %57 = fir.convert %54 : (i32) -> i64
      %58 = fir.convert %53 : (i32) -> i64
      %59 = fir.convert %52 : (i32) -> i64
      %c0_9 = arith.constant 0 : index
      %60 = fir.convert %59 : (i64) -> index
      %61 = arith.cmpi sgt, %60, %c0_9 : index
      %c0_10 = arith.constant 0 : index
      %62 = fir.convert %58 : (i64) -> index
      %63 = arith.cmpi sgt, %62, %c0_10 : index
      %c0_11 = arith.constant 0 : index
      %64 = fir.convert %57 : (i64) -> index
      %65 = arith.cmpi sgt, %64, %c0_11 : index
      %c0_12 = arith.constant 0 : index
      %66 = fir.convert %56 : (i64) -> index
      %67 = arith.cmpi sgt, %66, %c0_12 : index
      %68 = arith.select %67, %66, %c0_12 : index
      %69 = arith.select %65, %64, %c0_11 : index
      %70 = arith.select %63, %62, %c0_10 : index
      %71 = arith.select %61, %60, %c0_9 : index
      %72:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %73:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %74:2 = hlfir.declare %arg8 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %75 = fir.shape %71, %70 : (index, index) -> !fir.shape<2>
      %76:2 = hlfir.declare %arg9(%75) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
      %77 = fir.shape %69, %68 : (index, index) -> !fir.shape<2>
      %78:2 = hlfir.declare %arg10(%77) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf64>>, !fir.ref<!fir.array<?x?xf64>>)
      // DL-ASSIGN-NOT: omp.teams
      omp.teams {

        // DL-ASSIGN-NOT: omp.parallel
        // DL-ASSIGN-NOT: @_QFFrun_benchmarkEi_private_i32
        // DL-ASSIGN-NOT: @_QFFrun_benchmarkEi_private_j32
        omp.parallel private(@_QFFrun_benchmarkEi_private_i32 %72#0 -> %arg15, @_QFFrun_benchmarkEj_private_i32 %74#0 -> %arg16 : !fir.ref<i32>, !fir.ref<i32>) {
          %79:2 = hlfir.declare %arg15 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          %80:2 = hlfir.declare %arg16 {uniq_name = "_QFFrun_benchmarkEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          
          // DL-ASSIGN-NOT: omp.distribute
          // DL-ASSIGN-NOT: omp.wsloop
          // DL-ASSIGN-NOT: omp.loop_nest
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg17) : i32 = (%arg3) to (%arg4) inclusive step (%arg5) {
                hlfir.assign %arg17 to %79#0 : i32, !fir.ref<i32>
                %c1_i32_13 = arith.constant 1 : i32
                %81 = fir.convert %c1_i32_13 : (i32) -> index
                %82 = fir.load %73#0 : !fir.ref<i32>
                %83 = fir.convert %82 : (i32) -> index
                %c1_14 = arith.constant 1 : index
                %84 = fir.convert %81 : (index) -> i32
                %85 = fir.do_loop %arg18 = %81 to %83 step %c1_14 iter_args(%arg19 = %84) -> (i32) {
                  fir.store %arg19 to %80#0 : !fir.ref<i32>
                  %86 = fir.load %79#0 : !fir.ref<i32>
                  %87 = fir.convert %86 : (i32) -> i64
                  %88 = fir.load %80#0 : !fir.ref<i32>
                  %89 = fir.convert %88 : (i32) -> i64
                  %90 = hlfir.designate %78#0 (%87, %89)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
                  %91 = fir.load %90 : !fir.ref<f64>
                  %92 = fir.load %79#0 : !fir.ref<i32>
                  %93 = fir.convert %92 : (i32) -> i64
                  %94 = fir.load %80#0 : !fir.ref<i32>
                  %95 = fir.convert %94 : (i32) -> i64
                  %96 = hlfir.designate %76#0 (%93, %95)  : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
                  hlfir.assign %91 to %96 : f64, !fir.ref<f64>
                  %97 = fir.convert %c1_14 : (index) -> i32
                  %98 = fir.load %80#0 : !fir.ref<i32>
                  %99 = arith.addi %98, %97 overflow<nsw> : i32
                  fir.result %99 : i32
                }
                fir.store %85 to %80#0 : !fir.ref<i32>
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    return
  }
}

//--- single-loop-reduction.mlir
module {
  omp.private {type = private} @_QFFrun_benchmarkEi_private_i32 : i32
  omp.declare_reduction @add_reduction_f64 : f64 init {
  ^bb0(%arg0: f64):
    %cst = arith.constant 0.000000e+00 : f64
    omp.yield(%cst : f64)
  } combiner {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.addf %arg0, %arg1 fastmath<contract> : f64
    omp.yield(%0 : f64)
  }
  // SL-REDUCTION-LABEL: func.func private @_QFPrun_benchmark(
  func.func private @_QFPrun_benchmark(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.alloca i32
    %1 = fir.dummy_scope : !fir.dscope
    %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFFrun_benchmarkEi"}
    %3:2 = hlfir.declare %2 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4:2 = hlfir.declare %arg2 dummy_scope %1 arg 3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFFrun_benchmarkEsum"}
    %6:2 = hlfir.declare %5 {uniq_name = "_QFFrun_benchmarkEsum"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %7 = fir.load %4#0 : !fir.ref<i32>
    fir.store %7 to %0 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> i64
    %9 = fir.convert %8 : (i64) -> index
    %c0 = arith.constant 0 : index
    %10 = arith.cmpi sgt, %9, %c0 : index
    %11 = arith.select %10, %9, %c0 : index
    %12 = fir.shape %11 : (index) -> !fir.shape<1>
    %13:2 = hlfir.declare %arg0(%12) dummy_scope %1 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %14 = fir.load %4#0 : !fir.ref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = fir.convert %15 : (i64) -> index
    %c0_0 = arith.constant 0 : index
    %17 = arith.cmpi sgt, %16, %c0_0 : index
    %18 = arith.select %17, %16, %c0_0 : index
    %19 = fir.shape %18 : (index) -> !fir.shape<1>
    %20:2 = hlfir.declare %arg1(%19) dummy_scope %1 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %cst = arith.constant 0.000000e+00 : f64
    hlfir.assign %cst to %6#0 : f64, !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %21 = fir.load %4#0 : !fir.ref<i32>
    %c1_i32_1 = arith.constant 1 : i32
    %22 = omp.map.info var_ptr(%6#1 : !fir.ref<f64>, f64) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f64> {name = "sum"}
    %23 = omp.map.info var_ptr(%3#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
    %24 = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "n"}
    %c1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %25 = arith.subi %11, %c1 : index
    %26 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%25 : index) extent(%11 : index) stride(%c1 : index) start_idx(%c1 : index)
    %27 = omp.map.info var_ptr(%13#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%26) -> !fir.ref<!fir.array<?xf64>> {name = "x"}
    %28 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    // SL-REDUCTION: omp.target 
    // SL-REDUCTION-NOT: host_eval
    // SL-REDUCTION-SAME: map_entries
    omp.target host_eval(%c1_i32 -> %arg3, %21 -> %arg4, %c1_i32_1 -> %arg5 : i32, i32, i32) map_entries(%22 -> %arg6, %23 -> %arg7, %24 -> %arg8, %27 -> %arg9, %28 -> %arg10 : !fir.ref<f64>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<?xf64>>, !fir.ref<i32>) {
      %40 = fir.load %arg10 : !fir.ref<i32>
      %41 = fir.convert %40 : (i32) -> i64
      %c0_3 = arith.constant 0 : index
      %42 = fir.convert %41 : (i64) -> index
      %43 = arith.cmpi sgt, %42, %c0_3 : index
      %44 = arith.select %43, %42, %c0_3 : index
      %45:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEsum"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
      %46:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %47:2 = hlfir.declare %arg8 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %48 = fir.shape %44 : (index) -> !fir.shape<1>
      %49:2 = hlfir.declare %arg9(%48) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      // SL-REDUCTION-NOT: omp.teams  
      // SL-REDUCTION-NOT: reduction 
      // SL-REDUCTION-NOT: private
      omp.teams reduction(@add_reduction_f64 %45#0 -> %arg11 : !fir.ref<f64>) {
        %50:2 = hlfir.declare %arg11 {uniq_name = "_QFFrun_benchmarkEsum"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
        omp.parallel private(@_QFFrun_benchmarkEi_private_i32 %46#0 -> %arg12 : !fir.ref<i32>) {
          %51:2 = hlfir.declare %arg12 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          omp.distribute {
            omp.wsloop reduction(@add_reduction_f64 %50#0 -> %arg13 : !fir.ref<f64>) {
              omp.loop_nest (%arg14) : i32 = (%arg3) to (%arg4) inclusive step (%arg5) {
                %52:2 = hlfir.declare %arg13 {uniq_name = "_QFFrun_benchmarkEsum"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
                hlfir.assign %arg14 to %51#0 : i32, !fir.ref<i32>
                %53 = fir.load %52#0 : !fir.ref<f64>
                %54 = fir.load %51#0 : !fir.ref<i32>
                %55 = fir.convert %54 : (i32) -> i64
                %56 = hlfir.designate %49#0 (%55)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
                %57 = fir.load %56 : !fir.ref<f64>
                %58 = arith.addf %53, %57 fastmath<contract> : f64
                hlfir.assign %58 to %52#0 : f64, !fir.ref<f64>
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      } {omp.combined}
      omp.terminator
    } {omp.combined}
    %c6_i32 = arith.constant 6 : i32
    %29 = fir.address_of(@_QQclX093479350fcb2154ec223f299460e035) : !fir.ref<!fir.char<1,77>>
    %30 = fir.convert %29 : (!fir.ref<!fir.char<1,77>>) -> !fir.ref<i8>
    %c42_i32 = arith.constant 42 : i32
    %31 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %30, %c42_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %32 = fir.address_of(@_QQclX73756D3A20) : !fir.ref<!fir.char<1,5>>
    %c5 = arith.constant 5 : index
    %33:2 = hlfir.declare %32 typeparams %c5 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX73756D3A20"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
    %34 = fir.convert %33#0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
    %35 = fir.convert %c5 : (index) -> i64
    %36 = fir.call @_FortranAioOutputAscii(%31, %34, %35) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64) -> i1
    %37 = fir.load %6#0 : !fir.ref<f64>
    %38 = fir.call @_FortranAioOutputReal64(%31, %37) fastmath<contract> : (!fir.ref<i8>, f64) -> i1
    %39 = fir.call @_FortranAioEndIoStatement(%31) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
}
