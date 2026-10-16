; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/wide.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/vector.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR
; CALL: calls, inline assembly and unbound intrinsics are forbidden in bound scope
; VECTOR: LLVM vector values are forbidden in bound scope

;--- wide.ll
declare i128 @llvm.umax.i128(i128, i128)
define i128 @wide_minmax(i128 %a, i128 %b) "trex-bosc-bound-rvv"="1" {
  %maximum = call i128 @llvm.umax.i128(i128 %a, i128 %b)
  ret i128 %maximum
}

;--- vector.ll
declare <2 x i64> @llvm.umax.v2i64(<2 x i64>, <2 x i64>)
define <2 x i64> @vector_minmax(<2 x i64> %a, <2 x i64> %b)
    "trex-bosc-bound-rvv"="1" {
  %maximum = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %a, <2 x i64> %b)
  ret <2 x i64> %maximum
}
