; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
;
; Scalar f32 maximum-number uses the official intrinsic and real fmax.s.
; That instruction owns RISC-V +0/-0, sole/two-NaN and sNaN invalid-flag
; semantics. Comparison/select expansion must not raise invalid for qNaN.
declare float @llvm.maximumnum.f32(float, float)

; CHECK-LABEL: scalar_maximum:
; CHECK: flw
; CHECK: flw
; CHECK-NOT: flt.s
; CHECK-NOT: feq.s
; CHECK-NOT: call
; CHECK: fmax.s
; CHECK: fsw
; CHECK: ret
define void @scalar_maximum(ptr %lhs, ptr %rhs, ptr %output)
    "trex-bosc-bound-rvv"="1" {
  %a = load float, ptr %lhs
  %b = load float, ptr %rhs
  %maximum = call float @llvm.maximumnum.f32(float %a, float %b)
  store float %maximum, ptr %output
  ret void
}
