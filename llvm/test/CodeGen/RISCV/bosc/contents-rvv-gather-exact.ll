; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -riscv-v-vector-bits-min=512 -riscv-v-vector-bits-max=512 -filetype=obj -verify-machineinstrs %s -o %t.o

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)

; With exact VLEN=512, index16 is out of range and writes zero without reading
; any source lane. The old destination still carries the VL=0 preservation.
; CHECK-LABEL: gather_index16_zero:
; CHECK: vrgather.vi {{.*}}, 16
; CHECK: vse32.v
define void @gather_index16_zero(ptr %out, i64 %n) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %zero = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> poison, i64 16, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %zero, ptr %out, i64 %vl)
  ret void
}
