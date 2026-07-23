// RUN: split-file %s %t
// RUN: fir-opt %t/implicit-first-private.mlir --materialize-privates | FileCheck %s --check-prefix=IMPLICIT-FIRST-PRIVATE

//--- implicit-first-private.mlir
// IMPLICIT-FIRST-PRIVATE-LABEL: func.func private @_QFPrun_benchmark(
module  {
  omp.private {type = private} @_QFFrun_benchmarkEi_private_i32 : i32
  func.func private @_QFPrun_benchmark(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.alloca i32
    %1 = fir.alloca i32
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFFrun_benchmarkEi"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5:2 = hlfir.declare %arg2 dummy_scope %2 arg 3 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %6 = fir.alloca f64 {bindc_name = "temp", uniq_name = "_QFFrun_benchmarkEtemp"}
    %7:2 = hlfir.declare %6 {uniq_name = "_QFFrun_benchmarkEtemp"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    %8 = fir.load %5#0 : !fir.ref<i32>
    fir.store %8 to %1 : !fir.ref<i32>
    %9 = fir.convert %8 : (i32) -> i64
    %10 = fir.convert %9 : (i64) -> index
    %c0 = arith.constant 0 : index
    %11 = arith.cmpi sgt, %10, %c0 : index
    %12 = arith.select %11, %10, %c0 : index
    %13 = fir.shape %12 : (index) -> !fir.shape<1>
    %14:2 = hlfir.declare %arg0(%13) dummy_scope %2 arg 1 {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %15 = fir.load %5#0 : !fir.ref<i32>
    fir.store %15 to %0 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.convert %16 : (i64) -> index
    %c0_0 = arith.constant 0 : index
    %18 = arith.cmpi sgt, %17, %c0_0 : index
    %19 = arith.select %18, %17, %c0_0 : index
    %20 = fir.shape %19 : (index) -> !fir.shape<1>
    %21:2 = hlfir.declare %arg1(%20) dummy_scope %2 arg 2 {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
    %cst = arith.constant 1.200000e+01 : f64
    hlfir.assign %cst to %7#0 : f64, !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %22 = fir.load %5#0 : !fir.ref<i32>
    %c1_i32_1 = arith.constant 1 : i32
    %23 = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
    %24 = omp.map.info var_ptr(%5#1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = "n"}
    %c1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %25 = arith.subi %19, %c1 : index
    %26 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%25 : index) extent(%19 : index) stride(%c1 : index) start_idx(%c1 : index)
    %27 = omp.map.info var_ptr(%21#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%26) -> !fir.ref<!fir.array<?xf64>> {name = "y"}
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %28 = arith.subi %12, %c1_3 : index
    %29 = omp.map.bounds lower_bound(%c0_4 : index) upper_bound(%28 : index) extent(%12 : index) stride(%c1_3 : index) start_idx(%c1_3 : index)
    %30 = omp.map.info var_ptr(%14#1 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(implicit, tofrom) capture(ByRef) bounds(%29) -> !fir.ref<!fir.array<?xf64>> {name = "x"}
    %31 = omp.map.info var_ptr(%7#1 : !fir.ref<f64>, f64) map_clauses(implicit) capture(ByCopy) -> !fir.ref<f64> {name = "temp"}
    %32 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %33 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    // IMPLICIT-FIRST-PRIVATE: omp.target
    omp.target host_eval(%c1_i32 -> %arg3, %22 -> %arg4, %c1_i32_1 -> %arg5 : i32, i32, i32) map_entries(%23 -> %arg6, %24 -> %arg7, %27 -> %arg8, %30 -> %arg9, %31 -> %arg10, %32 -> %arg11, %33 -> %arg12 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>, !fir.ref<f64>, !fir.ref<i32>, !fir.ref<i32>) {
    // temp: 
    // IMPLICIT-FIRST-PRIVATE-DAG: %[[LOCAL:[^ ]+]] = fir.alloca
    // IMPLICIT-FIRST-PRIVATE-DAG: %[[PVALUE:[^ ]+]] = fir.load %arg10
    // IMPLICIT-FIRST-PRIVATE-DAG: fir.store %[[PVALUE]] to %[[LOCAL]]

    // i is also implicit private:
    // IMPLICIT-FIRST-PRIVATE-DAG: %[[LOCAL2:[^ ]+]] = fir.alloca
    // IMPLICIT-FIRST-PRIVATE-DAG: %[[PVALUE2:[^ ]+]] = fir.load %arg6
    // IMPLICIT-FIRST-PRIVATE-DAG: fir.store %[[PVALUE2]] to %[[LOCAL2]]


    // IMPLICIT-FIRST-PRIVATE-NOT: %arg6
    // IMPLICIT-FIRST-PRIVATE-NOT: %arg10
      %34 = fir.load %arg12 : !fir.ref<i32>
      %35 = fir.load %arg11 : !fir.ref<i32>
      %36 = fir.convert %35 : (i32) -> i64
      %37 = fir.convert %34 : (i32) -> i64
      %c0_5 = arith.constant 0 : index
      %38 = fir.convert %37 : (i64) -> index
      %39 = arith.cmpi sgt, %38, %c0_5 : index
      %c0_6 = arith.constant 0 : index
      %40 = fir.convert %36 : (i64) -> index
      %41 = arith.cmpi sgt, %40, %c0_6 : index
      %42 = arith.select %41, %40, %c0_6 : index
      %43 = arith.select %39, %38, %c0_5 : index
      %44:2 = hlfir.declare %arg6 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %45:2 = hlfir.declare %arg7 {uniq_name = "_QFFrun_benchmarkEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %46 = fir.shape %43 : (index) -> !fir.shape<1>
      %47:2 = hlfir.declare %arg8(%46) {uniq_name = "_QFFrun_benchmarkEy"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      %48 = fir.shape %42 : (index) -> !fir.shape<1>
      %49:2 = hlfir.declare %arg9(%48) {uniq_name = "_QFFrun_benchmarkEx"} : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf64>>, !fir.ref<!fir.array<?xf64>>)
      %50:2 = hlfir.declare %arg10 {uniq_name = "_QFFrun_benchmarkEtemp"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
      omp.teams {
        omp.parallel private(@_QFFrun_benchmarkEi_private_i32 %44#0 -> %arg13 : !fir.ref<i32>) {
          %51:2 = hlfir.declare %arg13 {uniq_name = "_QFFrun_benchmarkEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg14) : i32 = (%arg3) to (%arg4) inclusive step (%arg5) {
                hlfir.assign %arg14 to %51#0 : i32, !fir.ref<i32>
                %52 = fir.load %51#0 : !fir.ref<i32>
                %53 = fir.convert %52 : (i32) -> i64
                %54 = hlfir.designate %49#0 (%53)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
                %55 = fir.load %54 : !fir.ref<f64>
                %56 = fir.load %50#0 : !fir.ref<f64>
                %57 = arith.addf %55, %56 fastmath<contract> : f64
                %58 = fir.load %51#0 : !fir.ref<i32>
                %59 = fir.convert %58 : (i32) -> i64
                %60 = hlfir.designate %47#0 (%59)  : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
                hlfir.assign %57 to %60 : f64, !fir.ref<f64>
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
