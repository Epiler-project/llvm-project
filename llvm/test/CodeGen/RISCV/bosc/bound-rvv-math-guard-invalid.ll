; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/return.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TRAP
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/continue.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TRAP
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/other-intrinsic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; TRAP: trap must be immediately followed by unreachable
; CALL: calls, inline assembly and unbound intrinsics are forbidden in bound scope

;--- return.ll
declare void @llvm.trap()
define void @trap_return() "trex-bosc-bound-rvv"="1" {
  call void @llvm.trap()
  ret void
}

;--- continue.ll
declare void @llvm.trap()
define void @trap_continue(ptr %output) "trex-bosc-bound-rvv"="1" {
  call void @llvm.trap()
  store i32 0, ptr %output
  unreachable
}

;--- call.ll
declare void @external()
define void @external_call() "trex-bosc-bound-rvv"="1" {
  call void @external()
  ret void
}

;--- other-intrinsic.ll
declare double @llvm.sqrt.f64(double)
define double @unbound_math(double %value) "trex-bosc-bound-rvv"="1" {
  %result = call double @llvm.sqrt.f64(double %value)
  ret double %result
}
