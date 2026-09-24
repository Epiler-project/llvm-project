; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/invalid.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/invalid-concat.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONCAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/bad-seed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONCAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/bad-step.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONCAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/exit-use.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONCAT
; CHECK-LABEL: two_halves:
; CHECK: sw
; CHECK: {{b(ne|lt|ge)}}
; CHECK: ret
; INVALID: PACR field last must be proven in [0, 1]
; CONCAT: PACR field concat must be proven in [0, 7]

; The header PHI also reaches the exit with value 2. Its context-free range
; includes 2, but the instruction-port use is dominated by the loop guard.
;--- valid.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @two_halves() "tensix-executor"="trisc2" {
entry:
  br label %header
header:
  %half = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  %concat = sub i32 1, %half
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 %half, i32 0, i32 0, i32 %concat, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %next = add i32 %half, 1
  br label %header
exit:
  ret void
}

; A signed upper guard alone admits negative values. The verifier must keep
; the unsigned lower bound and must not reuse an unrelated body's guard.
;--- invalid.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @negative_half(i32 noundef %half) "tensix-executor"="trisc2" {
entry:
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 %half, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  br label %exit
exit:
  ret void
}

;--- invalid-concat.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @negative_concat(i32 noundef %half) "tensix-executor"="trisc2" {
entry:
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  %concat = sub i32 1, %half
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 0, i32 0, i32 0, i32 %concat, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  br label %exit
exit:
  ret void
}

;--- bad-seed.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @bad_seed() "tensix-executor"="trisc2" {
entry:
  br label %header
header:
  %half = phi i32 [ 0, %entry ], [ %next, %body ]
  %field = phi i32 [ 8, %entry ], [ %field.next, %body ]
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 0, i32 0, i32 0, i32 %field, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %next = add i32 %half, 1
  %field.next = add i32 %field, -1
  br label %header
exit:
  ret void
}

;--- bad-step.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @bad_step() "tensix-executor"="trisc2" {
entry:
  br label %header
header:
  %half = phi i32 [ 0, %entry ], [ %next, %body ]
  %field = phi i32 [ 1, %entry ], [ %field.next, %body ]
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 0, i32 0, i32 0, i32 %field, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %next = add i32 %half, 1
  %field.next = add i32 %field, -2
  br label %header
exit:
  ret void
}

;--- exit-use.ll
declare void @llvm.riscv.tt.pacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @exit_use() "tensix-executor"="trisc2" {
entry:
  br label %header
header:
  %half = phi i32 [ 0, %entry ], [ %next, %body ]
  %field = phi i32 [ 1, %entry ], [ %field.next, %body ]
  %active = icmp slt i32 %half, 2
  br i1 %active, label %body, label %exit
body:
  %next = add i32 %half, 1
  %field.next = add i32 %field, -1
  br label %header
exit:
  call void @llvm.riscv.tt.pacr.port(i32 0, i32 0, i32 0, i32 0, i32 %field, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}
