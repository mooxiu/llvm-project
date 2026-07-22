// RUN: split-file %s %t
// RUN: fir-opt %t/trivial-assign.mlir --flatten-target | FileCheck %s --check-prefix=TRIVIAL-ASSIGN
// RUN: fir-opt %t/single-loop-assign.mlir --flatten-target | FileCheck %s --check-prefix=SL-ASSIGN

//--- trivial-assign.mlir
// TRIVIAL-ASSIGN-LABEL: func.func private @_QFPrun_benchmark
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

//--- single-loop-assign.mlir
// SL-ASSIGN-LABEL: func.func private @_QFPrun_benchmark
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
// SL-ASSIGN-NOT: omp.loop_nest
// SL-ASSIGN-NOT: omp.
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

