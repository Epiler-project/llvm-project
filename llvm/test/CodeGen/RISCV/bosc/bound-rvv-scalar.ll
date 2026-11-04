; RUN: llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+m,+a,+f,+d,+c -O2 -verify-machineinstrs < %s | FileCheck %s

; The BOSC verifier is installed for all RISC-V functions. Querying exact VLEN
; is only valid on subtargets that actually have vector instructions.
; CHECK-LABEL: scalar:
; CHECK: add a0, a0, a1
; CHECK: ret
define i64 @scalar(i64 %a, i64 %b) {
  %sum = add i64 %a, %b
  ret i64 %sum
}
