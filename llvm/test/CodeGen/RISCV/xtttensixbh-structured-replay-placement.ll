; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; No input provides a word count. The fixed two-word body exactly fills slots
; 30..31. An implementation treating identity 200 as a slot, using a provisional
; length of one, or counting the end marker fails its final tuple/encoding.
; Automatic placement is checked for consistent legal placement, not slot 0.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @fixed_final_count() "tensix-executor"="trisc1" {
; FINAL-LABEL: name: fixed_final_count
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 2, 30,
; FINAL-NEXT: PseudoTTSFPURecordWord
; FINAL-NEXT: PseudoTTSFPURecordWord
; FINAL-NEXT: PseudoTTReplayRecordEnd
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 2, 30,
; FINAL-SAME: implicit $tt_l1
; FINAL-SAME: implicit-def $tt_l0
; FINAL-SAME: implicit-def $tt_l2
; ASM-LABEL: fixed_final_count:
; Architectural header: 0x04000000 | (30 << 14) | (2 << 4) | 1;
; direct-issue encoding rotates left by two, producing 0x101e0084.
; ASM: .word 0x101e0084
; ASM-NEXT: .word 0xf0000409
; ASM-NEXT: .word 0xf0000489
; ASM: .word 0x101e0080
; ASM: .word 0xc8100001
; ASM: .word 0xc8900021
; ASM: ret
  call void @llvm.riscv.tt.replay.template.begin(i32 200, i32 0, i32 1, i32 30)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 200)
  call void @llvm.riscv.tt.replay.template.execute(i32 200)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  ret void
}

define void @automatic_final_count() "tensix-executor"="trisc1" {
; FINAL-LABEL: name: automatic_final_count
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 2, [[START:[0-9]+]],
; FINAL-NEXT: PseudoTTSFPURecordWord
; FINAL-NEXT: PseudoTTSFPURecordWord
; FINAL-NEXT: PseudoTTReplayRecordEnd
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 2, [[START]],
; FINAL-SAME: implicit $tt_l1
; FINAL-SAME: implicit-def $tt_l0
; FINAL-SAME: implicit-def $tt_l2
; ASM-LABEL: automatic_final_count:
; ASM: .word 0xf0000409
; ASM-NEXT: .word 0xf0000489
; ASM: .word 0xc8100001
; ASM: .word 0xc8900021
; ASM: ret
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.replay.template.execute(i32 901)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  ret void
}

; Both templates remain live across the second recording. Automatic placement
; must avoid fixed slot 0, even though the second template executes first.
define void @automatic_avoids_live_fixed() "tensix-executor"="trisc1" {
; FINAL-LABEL: name: automatic_avoids_live_fixed
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 1, 0,
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 1, [[AUTO:[1-9][0-9]*]],
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, [[AUTO]],
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, 0,
  call void @llvm.riscv.tt.replay.template.begin(i32 100, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 100)
  call void @llvm.riscv.tt.replay.template.begin(i32 101, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 101)
  call void @llvm.riscv.tt.replay.template.execute(i32 101)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 100)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

; The first template has no use after the second recording. Identical fixed
; placement is legal here; rejecting every lexical overlap loses slot reuse.
define void @disjoint_fixed_lifetimes() "tensix-executor"="trisc1" {
; FINAL-LABEL: name: disjoint_fixed_lifetimes
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, 31,
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, 31,
  call void @llvm.riscv.tt.replay.template.begin(i32 110, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 110)
  call void @llvm.riscv.tt.replay.template.execute(i32 110)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 111, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 111)
  call void @llvm.riscv.tt.replay.template.execute(i32 111)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  ret void
}
