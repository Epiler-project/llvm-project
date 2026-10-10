; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/width_boundary.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/exact_discard.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/ashr_exact_discard.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/nuw_overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/nsw_overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison_value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison_count.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; Bounded dynamic shift counts are safe for defined operands. Keep all real
; induction backedges; no unrolling or freeze is needed. Flags remain proof
; obligations, and masking a poisoned result never makes it admissible.
; CHECK-LABEL: lshr_lut:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; CHECK-LABEL: ashr_lut:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; CHECK-LABEL: shl_bounded:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; POISON: MVMUL field addr_mode must be defined and non-poison

;--- valid.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)

define void @lshr_lut() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %masked = and i32 %issue, 7
  %count = shl i32 %masked, 2
  %shifted = lshr i32 1344282640, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

define void @ashr_lut() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %masked = and i32 %issue, 7
  %count = shl i32 %masked, 2
  %shifted = ashr i32 -803201008, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

define void @shl_bounded() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = shl i32 1, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- width_boundary.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @width_boundary() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %masked = and i32 %issue, 7
  %count = mul i32 %masked, 8
  %shifted = lshr i32 1344282640, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- exact_discard.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @exact_discard() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = lshr exact i32 3, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- ashr_exact_discard.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @ashr_exact_discard() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = ashr exact i32 -3, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- nuw_overflow.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @nuw_overflow() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = shl nuw i32 -2147483648, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- nsw_overflow.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @nsw_overflow() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = shl nsw i32 1073741824, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- poison_value.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @poison_value() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 %issue, 7
  %shifted = lshr i32 poison, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}

;--- poison_count.ll
declare void @llvm.riscv.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)
define void @poison_count() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %issue = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %issue, 16
  br i1 %active, label %body, label %exit
body:
  %count = and i32 poison, 7
  %shifted = lshr i32 1344282640, %count
  %mode = and i32 %shifted, 7
  call void @llvm.riscv.tt.mvmul.port(i32 0, i32 0, i32 %mode, i32 0, i32 0)
  %next = add i32 %issue, 1
  br label %header
exit:
  ret void
}
