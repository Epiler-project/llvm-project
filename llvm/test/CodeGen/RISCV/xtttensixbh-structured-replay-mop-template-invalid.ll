; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/slot.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SLOT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/dominance.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DOMINANCE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DOMINANCE
;
; SLOT: immediate operand 1 must be in [2, 8]
; DOMINANCE: replay template execute requires a dominating record

;--- slot.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
define void @invalid_slot() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.replay.template.begin(i32 71, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 71)
  call void @llvm.riscv.tt.replay.template.mop(i32 71, i32 9)
  ret void
}

;--- dominance.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
define void @conditional_record(i1 noundef %condition) "tensix-executor"="trisc0" {
entry:
  br i1 %condition, label %record, label %bind
record:
  call void @llvm.riscv.tt.replay.template.begin(i32 71, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 71)
  br label %bind
bind:
  call void @llvm.riscv.tt.replay.template.mop(i32 71, i32 3)
  ret void
}

;--- unknown.ll
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
define void @unknown_record() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.replay.template.mop(i32 71, i32 3)
  ret void
}
