// Nested (non-combined) parallelism inside target teams distribute:
// the outer omp.loop_nest must be flattened before the inner one.
// RUN: split-file %s %t
// RUN: fir-opt %t/nested-loop-nest.mlir --flatten-target | FileCheck %s --check-prefix=NESTED
// RUN: fir-opt %t/nested-loop-nest-reduction.mlir --flatten-target | FileCheck %s --check-prefix=NESTED-RED


//--- nested-loop-nest.mlir
// Simple: teams distribute over i, with a parallel do over j in its body.
//   do i = 1, 8          ! teams distribute
//     do j = 1, 8        ! parallel do
//       x(j, i) = 1.0
module {
  // NESTED-LABEL: func.func @nested
  func.func @nested(%arg0: !fir.ref<!fir.array<8x8xf64>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<8x8xf64>>, !fir.array<8x8xf64>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.array<8x8xf64>> {name = "x"}
    // NESTED: omp.target
    // NESTED-NOT: omp.teams
    // NESTED-NOT: omp.distribute
    // NESTED: scf.parallel
    // NESTED-NOT: omp.parallel
    // NESTED-NOT: omp.wsloop
    // NESTED: scf.parallel
    // NESTED: fir.coordinate_of
    // NESTED: fir.store
    // NESTED-NOT: omp.loop_nest
    // NESTED: omp.terminator
    omp.target map_entries(%0 -> %arg1 : !fir.ref<!fir.array<8x8xf64>>) {
      %c1 = arith.constant 1 : i32
      %c8 = arith.constant 8 : i32
      %cst = arith.constant 1.000000e+00 : f64
      omp.teams {
        omp.distribute {
          omp.loop_nest (%i) : i32 = (%c1) to (%c8) inclusive step (%c1) {
            omp.parallel {
              omp.wsloop {
                omp.loop_nest (%j) : i32 = (%c1) to (%c8) inclusive step (%c1) {
                  %ii = fir.convert %i : (i32) -> index
                  %jj = fir.convert %j : (i32) -> index
                  %p = fir.coordinate_of %arg1, %jj, %ii : (!fir.ref<!fir.array<8x8xf64>>, index, index) -> !fir.ref<f64>
                  fir.store %cst to %p : !fir.ref<f64>
                  omp.yield
                }
              }
              omp.terminator
            }
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}


//--- nested-loop-nest-reduction.mlir
// Complex: the structure of the RI-MP2 energy loop (RIMP2_ENERGYIJ), with a
// reduction on both levels.
//   !$omp target map(tofrom:e2)
//   !$omp teams distribute reduction(+:e2)
//   do ic = 1, 8
//     e2_t = 0
//     !$omp parallel do reduction(+:e2_t) collapse(2)
//     do ib = 1, 8
//       do ia = 1, 8
//         e2_t = e2_t + x(ia, ib)
//     e2 = e2 + e2_t
module {
  omp.declare_reduction @add_reduction_f64 : f64 init {
  ^bb0(%arg0: f64):
    %cst = arith.constant 0.000000e+00 : f64
    omp.yield(%cst : f64)
  } combiner {
  ^bb0(%arg0: f64, %arg1: f64):
    %0 = arith.addf %arg0, %arg1 fastmath<contract> : f64
    omp.yield(%0 : f64)
  }
  // NESTED-RED-LABEL: func.func @nested_reduction
  func.func @nested_reduction(%arg0: !fir.ref<!fir.array<8x8xf64>>, %arg1: !fir.ref<f64>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<8x8xf64>>, !fir.array<8x8xf64>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.array<8x8xf64>> {name = "x"}
    %1 = omp.map.info var_ptr(%arg1 : !fir.ref<f64>, f64) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f64> {name = "e2"}
    // NESTED-RED: omp.target map_entries(%{{[0-9]+}} -> %{{[a-z0-9]+}}, %{{[0-9]+}} -> [[E2:%[a-z0-9]+]] :
    // NESTED-RED: [[E2T:%[^ ]+]] = fir.alloca f64 {bindc_name = "e2_t"}
    // NESTED-RED-NOT: omp.teams
    // ic loop: teams reduction(+:e2) becomes the loop's reduction, with the
    // original e2 as init. Each iteration has its own copy of e2, starting at 0.
    // NESTED-RED: [[E2_INIT:%[^ ]+]] = fir.load [[E2]]
    // NESTED-RED: [[E2_RES:%[^ ]+]] = scf.parallel ({{.*}}) init ([[E2_INIT]]) -> f64 {
    // NESTED-RED-NEXT: [[TEAMS_ACC:%[^ ]+]] = fir.alloca f64
    // NESTED-RED: fir.store %{{.*}} to [[TEAMS_ACC]]
    // e2_t = 0, then the ib/ia loop (collapse(2): one 2-D loop) with
    // reduction(+:e2_t), init e2_t.
    // NESTED-RED: fir.store %{{.*}} to [[E2T]]
    // NESTED-RED: [[E2T_INIT:%[^ ]+]] = fir.load [[E2T]]
    // NESTED-RED: [[E2T_RES:%[^ ]+]] = scf.parallel (%{{[^,]+}}, %{{[^)]+}}) = {{.*}} init ([[E2T_INIT]]) -> f64 {
    // NESTED-RED-NEXT: [[WS_ACC:%[^ ]+]] = fir.alloca f64
    // NESTED-RED: fir.store %{{.*}} to [[WS_ACC]]
    // NESTED-RED: fir.load [[WS_ACC]]
    // NESTED-RED: arith.addf
    // NESTED-RED: fir.store %{{.*}} to [[WS_ACC]]
    // NESTED-RED: [[WS_X:%[^ ]+]] = fir.load [[WS_ACC]]
    // NESTED-RED-NEXT: scf.reduce([[WS_X]] : f64) {
    // NESTED-RED: arith.addf
    // NESTED-RED: scf.reduce.return
    // NESTED-RED: fir.store [[E2T_RES]] to [[E2T]]
    // e2 = e2 + e2_t updates this iteration's copy of e2, which is then reduced.
    // NESTED-RED: fir.load [[E2T]]
    // NESTED-RED: fir.load [[TEAMS_ACC]]
    // NESTED-RED: arith.addf
    // NESTED-RED: fir.store %{{.*}} to [[TEAMS_ACC]]
    // NESTED-RED: [[TEAMS_X:%[^ ]+]] = fir.load [[TEAMS_ACC]]
    // NESTED-RED-NEXT: scf.reduce([[TEAMS_X]] : f64) {
    // NESTED-RED: arith.addf
    // NESTED-RED: scf.reduce.return
    // NESTED-RED: fir.store [[E2_RES]] to [[E2]]
    // NESTED-RED-NOT: omp.loop_nest
    // NESTED-RED-NOT: reduction(
    // NESTED-RED: omp.terminator
    omp.target map_entries(%0 -> %x, %1 -> %e2 : !fir.ref<!fir.array<8x8xf64>>, !fir.ref<f64>) {
      %c1 = arith.constant 1 : i32
      %c8 = arith.constant 8 : i32
      %zero = arith.constant 0.000000e+00 : f64
      %e2t = fir.alloca f64 {bindc_name = "e2_t"}
      omp.teams reduction(@add_reduction_f64 %e2 -> %e2_acc : !fir.ref<f64>) {
        omp.distribute {
          omp.loop_nest (%ic) : i32 = (%c1) to (%c8) inclusive step (%c1) {
            fir.store %zero to %e2t : !fir.ref<f64>
            omp.parallel {
              omp.wsloop reduction(@add_reduction_f64 %e2t -> %e2t_acc : !fir.ref<f64>) {
                omp.loop_nest (%ib, %ia) : i32 = (%c1, %c1) to (%c8, %c8) inclusive step (%c1, %c1) collapse(2) {
                  %ia_idx = fir.convert %ia : (i32) -> index
                  %ib_idx = fir.convert %ib : (i32) -> index
                  %p = fir.coordinate_of %x, %ia_idx, %ib_idx : (!fir.ref<!fir.array<8x8xf64>>, index, index) -> !fir.ref<f64>
                  %v = fir.load %p : !fir.ref<f64>
                  %acc = fir.load %e2t_acc : !fir.ref<f64>
                  %sum = arith.addf %acc, %v fastmath<contract> : f64
                  fir.store %sum to %e2t_acc : !fir.ref<f64>
                  omp.yield
                }
              }
              omp.terminator
            }
            %t = fir.load %e2t : !fir.ref<f64>
            %e = fir.load %e2_acc : !fir.ref<f64>
            %s = arith.addf %e, %t fastmath<contract> : f64
            fir.store %s to %e2_acc : !fir.ref<f64>
            omp.yield
          }
        }
        omp.terminator
      }
      omp.terminator
    }
    return
  }
}
