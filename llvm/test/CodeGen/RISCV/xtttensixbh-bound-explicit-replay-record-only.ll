; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: FileCheck %s < %t.o0.s
; RUN: FileCheck %s < %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o2.mir
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o0.s %t.o0.mir record_only
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o2.s %t.o2.mir record_only
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=riscv-tensix-explicit-replay %s -o - | FileCheck %s --check-prefix=NORMALIZED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -stop-after=riscv-tensix-explicit-replay %s -o - | FileCheck %s --check-prefix=NORMALIZED
;
; Record-only captures the instruction, not its register values, and does not
; execute its L0 write. First store must still observe the authored C10 value.
; Changing L1 after recording changes what the later execute writes to L0.
; The MIR check deliberately names no proposed private payload/replay pseudo:
; recorded words must have only issue effects, and execution must read L1 and
; define L0. Final words are checked independently of that representation.

declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @record_only() "tensix-executor"="trisc1" {
; NORMALIZED-LABEL: name: record_only
; NORMALIZED: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; NORMALIZED-NEXT: PseudoTTSFPURecordWord
; NORMALIZED-NEXT: PseudoTTReplayRecordEnd
; NORMALIZED: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; NORMALIZED-SAME: implicit $tt_l1
; NORMALIZED-SAME: implicit-def $tt_l0
; CHECK-LABEL: record_only:
; CHECK: .word 0xf0002809
; CHECK: .word 0xf0000849
; Slot5/length1/(load1,execute0), distinct from record-and-execute bit3.
; CHECK: .word 0x10050044
; CHECK-NEXT: .word 0xf0000409
; CHECK: .word 0xc8100001
; CHECK: .word 0xf0002449
; CHECK: .word 0x10050040
; CHECK: .word 0xc8100021
; CHECK: ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  ret void
}
