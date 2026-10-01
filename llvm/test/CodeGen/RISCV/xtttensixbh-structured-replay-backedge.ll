; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/conflict.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/conflict.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/rerecord.ll -o - | FileCheck %s --check-prefix=RECORD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/rerecord.ll -o - | FileCheck %s --check-prefix=RECORD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/rerecord.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/rerecord.ll -o /dev/null
;
; The last lexical execute is not the end of template lifetime: the real
; backedge reaches it again. Even an unused record overwrites a live slot.
; Conversely an authored rerecord on every iteration starts a new version;
; the earlier version need not survive across that same backedge.
; CONFLICT: structured replay fixed placement conflicts or exceeds capacity
; RECORD-LABEL: name: record_on_each_iteration
; RECORD: successors:
; RECORD: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; RECORD-NEXT: PseudoTTSFPURecordWord
; RECORD-NOT: {{(PseudoTT|TTSFP)}}
; RECORD: PseudoTTReplayRecordEnd
; RECORD: PseudoTTExplicitSFPUReplay 0, 0, 1, 31,
; RECORD: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; RECORD-NEXT: PseudoTTSFPURecordWord
; A scalar loop-counter update may be scheduled here. It is not a recorded
; Tensix word and does not change the one-word template or its real backedge.
; RECORD-NOT: {{(PseudoTT|TTSFP)}}
; RECORD: PseudoTTReplayRecordEnd
; RECORD-NOT: PseudoTTExplicitSFPUReplay
; RECORD: PseudoRET

;--- conflict.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

define void @unused_record_clobbers_next_iteration(i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 401, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 401)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.replay.template.execute(i32 401)
  call void @llvm.riscv.tt.replay.template.begin(i32 402, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 402)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

;--- rerecord.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

define void @record_on_each_iteration(i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.replay.template.begin(i32 403, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 403)
  call void @llvm.riscv.tt.replay.template.execute(i32 403)
  call void @llvm.riscv.tt.replay.template.begin(i32 404, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 404)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
