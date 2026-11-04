; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/f64.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/maximum.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: calls, inline assembly and unbound intrinsics are forbidden in bound scope
;
;--- f64.ll
declare double @llvm.maximumnum.f64(double, double)
define double @scalar_maximum_f64(double %a, double %b)
    "trex-bosc-bound-rvv"="1" {
  %maximum = call double @llvm.maximumnum.f64(double %a, double %b)
  ret double %maximum
}
;
;--- maximum.ll
declare float @llvm.maximum.f32(float, float)
define float @propagating_maximum(float %a, float %b)
    "trex-bosc-bound-rvv"="1" {
  %maximum = call float @llvm.maximum.f32(float %a, float %b)
  ret float %maximum
}
