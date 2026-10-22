; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O3 -verify-machineinstrs %s -o - | FileCheck %s

declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)

; Preserve the dynamic source loop and one pair of physical CC actions per
; iteration, without an SFPR temporary, predicate vector or replay expansion.
define void @compare_loop(i32 %count) "tensix-executor"="trisc1" {
; CHECK-LABEL: compare_loop:
; CHECK-NOT: .word
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK: .word 0x5c000406
; CHECK-NOT: .word
; CHECK: .word 0x58000406
; CHECK-NOT: .word
; CHECK: {{bnez?}} {{.*}}, [[LOOP]]
; CHECK-NOT: .word
; CHECK: ret
entry:
  br label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 1)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
