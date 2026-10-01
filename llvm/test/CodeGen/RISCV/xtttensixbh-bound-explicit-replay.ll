; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: FileCheck %s < %t.o0.s
; RUN: FileCheck %s < %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o2.mir
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o0.s %t.o0.mir record_and_execute replay_runtime_loop
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o2.s %t.o2.mir record_and_execute replay_runtime_loop
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=riscv-tensix-explicit-replay %s -o - | FileCheck %s --check-prefix=NORMALIZED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -stop-after=riscv-tensix-explicit-replay %s -o - | FileCheck %s --check-prefix=NORMALIZED
;
; Public bound physical operands and the existing ordinary REPLAY ABI only.
; Record-and-execute must execute its MOVAll immediately. A later execution
; reads current L1, not the value that L1 had when the word was recorded.
; Its source differs from record_only only in execute_while_loading: the
; first observation is incoming L2 here and preserved C10 there. C9/C10 are
; explicit architectural inputs, not compiler-created setup.
; The helper interprets the actual emitted words for this small instruction
; subset. This is software codegen evidence, not a hardware execution claim.

declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @record_and_execute() "tensix-executor"="trisc1" {
; NORMALIZED-LABEL: name: record_and_execute
; NORMALIZED: PseudoTTExplicitSFPUReplay 1, 1, 1, 5,
; NORMALIZED-NEXT: $tt_l0 = TTSFPMOVAll $tt_l1,
; NORMALIZED-NEXT: PseudoTTReplayRecordEnd
; NORMALIZED: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; NORMALIZED-SAME: implicit $tt_l1
; NORMALIZED-SAME: implicit-def $tt_l0
; CHECK-LABEL: record_and_execute:
; CHECK: .word 0xf0002809
; CHECK: .word 0xf0000849
; The raw REPLAY word is 0x04000000 | (start<<14) | (len<<4)
; | (execute_while_loading<<1) | load; direct issue rotates left by two.
; Slot5/length1/(load1,execute1) therefore emits 0x1005004c.
; CHECK: .word 0x1005004c
; CHECK-NEXT: .word 0xf0000409
; CHECK: .word 0xc8100001
; CHECK: .word 0xf0002449
; CHECK: .word 0x10050040
; CHECK: .word 0xc8100021
; CHECK: ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  ret void
}

; Recording occurs once. A runtime trip count executes the same recorded
; word on each real backedge. The L1 write follows execution and must not be
; hoisted before it: the first iteration observes C9, subsequent ones C10.
; The word observer checks exact static counts across the complete function,
; including any prologue/epilogue, so peeling/duplication also fails.
define void @replay_runtime_loop(i32 %count) "tensix-executor"="trisc1" {
; NORMALIZED-LABEL: name: replay_runtime_loop
; NORMALIZED: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; NORMALIZED-NEXT: PseudoTTSFPURecordWord
; NORMALIZED-NEXT: PseudoTTReplayRecordEnd
; NORMALIZED: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; NORMALIZED-SAME: implicit $tt_l1
; NORMALIZED-SAME: implicit-def $tt_l0
; CHECK-LABEL: replay_runtime_loop:
; CHECK: .word 0xf0002809
; CHECK: .word 0xf0002449
; CHECK: .word 0x10050044
; CHECK-NEXT: .word 0xf0000409
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK: .word 0x10050040
; CHECK: .word 0xf0002849
; CHECK: .word 0xc8100021
; CHECK: {{bne|bltu|blt|bnez}} {{.*}}[[LOOP]]
; CHECK: ret
entry:
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.record.end()
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
