; RUN: llc -mtriple=riscv32 -mattr=+m,+zba,+zbb,+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+zba,+zbb,+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=UNOPT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -stop-after=riscv-tensix-issue-word-fold %s -o - | FileCheck %s --check-prefix=FOLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %S/Inputs/xtttensixbh-private-issue-word.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRIVATE
;
; Ordinary instruction-port issue is encoded before selection. A field chosen
; by a select tree of constants becomes branches on its conditions, outermost
; first, each yielding a complete word; the loop-invariant port address is
; hoisted. The real 16-iteration loop remains; nothing is unrolled.

declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)

; The outermost select condition is tested first; each leaf is a whole word.
; FOLD-LABEL: define void @mvmul_modes(
; FOLD: [[EVEN:%[0-9]+]] = icmp eq i32 %{{[0-9]+}}, 0
; FOLD-NEXT: br i1 [[EVEN]], label %tt.word.t, label %tt.word.f
; FOLD: %tt.word = phi i32 [ 637534208, %tt.word.t ],
; FOLD-SAME: [ 637550592,
; FOLD-SAME: [ 637566976,
; FOLD-SAME: [ 637599744,
; FOLD-SAME: [ 637616128,
; FOLD: call void @llvm.riscv.tt.issue.word(i32 0, i32 38, i32 %tt.word)
; FOLD-NOT: = select i1
; FOLD-NOT: llvm.riscv.tt.mvmul.port

; CHECK-LABEL: mvmul_modes:
; CHECK: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NOT: lui {{[a-z0-9]+}}, 1048128
; 0x26000000: the even-issue word is one LUI, and is tested first.
; CHECK: lui [[WORD:[a-z0-9]+]], 155648
; CHECK: sw [[WORD]], 0([[PORT]])
; CHECK: andi [[BIT:[a-z0-9]+]], {{[a-z0-9]+}}, 1
; CHECK-NEXT: beqz [[BIT]],
; 0x26000000 | 4 << 14 is selected last.
; CHECK: lui [[WORD]], 155664
; CHECK-NOT: lui {{[a-z0-9]+}}, 1048128
; CHECK: ret
; UNOPT-LABEL: mvmul_modes:
; UNOPT: sw {{[a-z0-9]+}}, 0({{[a-z0-9]+}})
define void @mvmul_modes() "tensix-executor"="trisc1" {
entry:
  br label %loop
loop:
  %issue = phi i32 [ 0, %entry ], [ %next, %loop ]
  %end_first_half = icmp eq i32 %issue, 7
  %mode_end = select i1 %end_first_half, i32 4, i32 5
  %bits7 = and i32 %issue, 7
  %mode2_test = icmp eq i32 %bits7, 3
  %mode2 = select i1 %mode2_test, i32 2, i32 %mode_end
  %bits3 = and i32 %issue, 3
  %mode1_test = icmp eq i32 %bits3, 1
  %mode1 = select i1 %mode1_test, i32 1, i32 %mode2
  %bits1 = and i32 %issue, 1
  %mode0_test = icmp eq i32 %bits1, 0
  %mode = select i1 %mode0_test, i32 0, i32 %mode1
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add nuw nsw i32 %issue, 1
  %done = icmp eq i32 %next, 16
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; PRIVATE: compiler-private Tensix issue word in input
