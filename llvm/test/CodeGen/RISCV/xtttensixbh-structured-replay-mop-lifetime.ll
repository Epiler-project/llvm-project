; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/conflict.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/conflict.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/replace.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/replace.ll -o /dev/null
;
; CONFLICT: structured replay fixed placement conflicts or exceeds capacity
;
; A MOP slot retains its recorded identity across a real loop backedge. An
; unused recording must not clobber it after the last lexical MOP execution.
; In the replacement case, clearing/rebinding the slot kills that old demand;
; function-long pinning and an uncorrelated union of loop generations reject
; this legal fixed-slot reuse.

;--- conflict.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()

define void @mop_backedge_occupancy(i32 noundef %count) "tensix-executor"="trisc0" {
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 1, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 1, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.begin(i32 2, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 2)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

;--- replace.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()

define void @mop_generations_reuse(i32 noundef %count) "tensix-executor"="trisc0" {
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 1, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 1, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.replay.template.begin(i32 2, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 2, i32 3)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
