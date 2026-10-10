; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/assertion-only.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/undef-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-masked-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; InstCombine rewrites the Unpack face x-end expression (face & 1) * 16 + 15
; to a disjoint OR. Prove disjointness from the operands, then independently
; prove their definedness through the real loop backedge. The flag itself is
; not evidence, and a disjoint mask cannot repair a poisoned operand.
; CHECK-LABEL: unpack_face_ends:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; CHECK-LABEL: disjoint_dynamic_fields:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; POISON: SETADCXX field x_end2 must be defined and non-poison

;--- valid.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)

define void @unpack_face_ends() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp slt i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %shift = shl i32 %face, 4
  %begin = and i32 %shift, 16
  %end = or disjoint i32 %begin, 15
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 %begin, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

define void @disjoint_dynamic_fields() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %shift = shl i32 %face, 4
  %lo = and i32 %face, 3
  %hi = and i32 %shift, 48
  %end = or disjoint i32 %hi, %lo
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- overlap.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @overlap() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %bit = and i32 %face, 1
  %end = or disjoint i32 %bit, 1
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- assertion-only.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @assertion_only(i32 noundef %value) "tensix-executor"="trisc0" {
  %bounded = and i32 %value, 31
  %end = or disjoint i32 %bounded, 15
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- poison-input.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @poison_input() "tensix-executor"="trisc0" {
  %bounded = and i32 poison, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- undef-input.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @undef_input(i32 %value) "tensix-executor"="trisc0" {
  %bounded = and i32 %value, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- poison-masked-input.ll
declare void @llvm.riscv.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @poison_masked_input() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %overflow = shl nuw i32 %face, 31
  %bounded = and i32 %overflow, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.riscv.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}
