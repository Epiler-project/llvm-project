; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; UNKNOWN: read of undefined physical vector register v4 covering the active setvl lanes

; AVL>=32 guarantees VL=16 only for a target with exact VLEN=512. Normalize
; such a known configuration to the immediate form, preserving its actual VL
; result and enabling independent machine-prefix verification.
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; CHECK-LABEL: clamped:
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vse32.v v4
; CHECK: ret
define i64 @clamped(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 32)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret i64 %vl0
}
