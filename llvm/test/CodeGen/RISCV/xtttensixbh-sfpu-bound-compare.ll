; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM

declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)

; Comparing two already-bound sources must not write VD, insert a preserving
; copy, allocate an SFPR virtual, or create scalar scratch. Alias is legal.
define void @compare() "tensix-executor"="trisc1" {
; MIR-LABEL: name: compare
; MIR: registers: []
; MIR: TTSFPGT $tt_l0, $tt_l1, 1, implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; MIR-NEXT: TTSFPLE $tt_l7, $tt_c15, 1, implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; MIR-NEXT: TTSFPGT $tt_l3, $tt_l3, 1, implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; MIR-NOT: PseudoTTBound
; ASM-LABEL: compare:
; ASM: .word 0x5c000406
; ASM-NEXT: .word 0x58003dc6
; ASM-NEXT: .word 0x5c000cc6
; ASM-NEXT: ret
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfple(i32 7, i32 15, i32 1)
  call void @llvm.riscv.tt.bound.sfpgt(i32 3, i32 3, i32 1)
  ret void
}
