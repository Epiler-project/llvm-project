; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=PseudoTTBound --implicit-check-not=':sfpr'
; The bound groups remain one native word each. VC may alias an output.
declare void @llvm.riscv.tt.bound.sfpshft2.copy4(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpshft2.rotate4(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpshft2(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()

define void @shift_groups() "tensix-executor"="trisc1" {
; ASM-LABEL: shift_groups:
; ASM: .word 0x50000002
; ASM-NEXT: .word 0x50000006
; ASM-NEXT: .word 0x5000000a
; ASM-NEXT: .word 0x3c000002
; ASM-NEXT: .word 0x5000100a
; MIR-LABEL: name: shift_groups
; MIR: TTSFPSHFT2Copy4 0,
; MIR-SAME: implicit-def $tt_l0
; MIR-SAME: implicit-def $tt_l1
; MIR-SAME: implicit-def $tt_l2
; MIR-SAME: implicit-def $tt_l3
; MIR-SAME: implicit $tt_l0
; MIR-SAME: implicit $tt_l1
; MIR-SAME: implicit $tt_l2
; MIR-SAME: implicit $tt_l3
; MIR: TTSFPSHFT2Copy4 1,
; MIR: TTSFPSHFT2Rotate4 $tt_l0,
; MIR: TTSFPSHFT2Rotate4 $tt_l4,
  call void @llvm.riscv.tt.bound.sfpshft2.copy4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpshft2.copy4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 1)
  call void @llvm.riscv.tt.bound.sfpshft2.rotate4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpshft2.rotate4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 4)
  ret void
}

; A one-word record ending in the cross-lane mode must retain its cooldown
; at execution, even though the replay instruction itself has no static result.
define void @replay_shift_cooldown() "tensix-executor"="trisc1" {
; ASM-LABEL: replay_shift_cooldown:
; ASM: .word 0x10050040
; ASM-NEXT: .word 0x3c000002
; ASM-NEXT: .word 0xf0000189
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpshft2.rotate4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 0)
  ret void
}
