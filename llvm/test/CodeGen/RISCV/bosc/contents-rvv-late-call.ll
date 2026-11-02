; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs %s -o %t.s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs %s -o %t.s 2>&1 | FileCheck %s
; CHECK: machine call or inline assembly escapes contents scope

; A scalar operation may become a libcall after the IR receiver. Reject the
; late call even before any VR value is live, rather than relying on RA pressure.
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
define void @late_call_no_live(ptr %p, ptr %out, i128 %x, i128 %y) "trex-bosc-contents-rvv"="1" {
  %q = udiv i128 %x, %y
  store volatile i128 %q, ptr %out
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %v, ptr %out, i64 %vl)
  ret void
}
