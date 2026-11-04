; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs %s -o - | FileCheck %s

; The BOSC ingress verifier may add noimplicitfloat to a bound function.
; Its analysis contract must preserve the already-run stack protector even
; when the current function needs no policy change. Otherwise ISel rebuilds
; StackProtector after AA and invalidates the AA it just requested.
define i32 @scalar_increment(i32 %value) {
; CHECK-LABEL: scalar_increment:
; CHECK: addi{{w?}} a0, a0, 1
; CHECK: ret
  %next = add i32 %value, 1
  ret i32 %next
}
