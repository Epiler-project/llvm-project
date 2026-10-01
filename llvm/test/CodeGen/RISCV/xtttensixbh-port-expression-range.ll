; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/out-of-range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/negative-seed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/signed-guard.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/non-dominating-guard.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unsafe-shift.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
;
; SCEV bounds each operand of the bitwise OR, but treats the OR itself as an
; unknown. LVI does not establish the induction's nonnegative lower bound.
; Compose their facts through the actual scalar expression. The face order
; 0, 1, 2, 3 maps to output rows 0, 32, 16, 48, with real CFG backedges.
; CHECK-LABEL: transpose_faces:
; CHECK: sw
; CHECK: {{b(ne|lt|ge)}}
; CHECK: ret
; CHECK-LABEL: transpose_faces_with_rows:
; CHECK: sw
; CHECK: {{b(ne|lt|ge)}}
; CHECK: ret
; RANGE: SETADC field value must be proven in [0, 262143]
; DEFINED: SETADC field value must be defined and non-poison

;--- valid.ll
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
declare void @llvm.riscv.tt.unpacr.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)

define void @transpose_faces() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp slt i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

; Retain the nested unpack row loop and separate face latch from the authored
; target program. The proof must not require rotation or loop-body expansion.
define void @transpose_faces_with_rows() "tensix-executor"="trisc0" {
entry:
  br label %face.header
face.header:
  %face = phi i32 [ 0, %entry ], [ %face.next, %face.latch ]
  %face.active = icmp slt i32 %face, 4
  br i1 %face.active, label %face.body, label %exit
face.body:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  br label %row.header
row.header:
  %row = phi i32 [ 0, %face.body ], [ %row.next, %row.body ]
  %row.active = icmp slt i32 %row, 16
  br i1 %row.active, label %row.body, label %face.latch
row.body:
  call void @llvm.riscv.tt.unpacr.port(i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 68, i32 0)
  %row.next = add i32 %row, 1
  br label %row.header
face.latch:
  %face.next = add i32 %face, 1
  br label %face.header
exit:
  ret void
}

;--- out-of-range.ll
; The same expression reaches 262144 at face 32768. A syntactically similar
; induction or a multiplication with a constant is not sufficient evidence.
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
define void @out_of_range() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp slt i32 %face, 32769
  br i1 %active, label %body, label %exit
body:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- negative-seed.ll
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
define void @negative_seed() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ -1, %entry ], [ %next, %body ]
  %active = icmp slt i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- signed-guard.ll
; Negative values satisfy this guard. SE/LVI composition must not invent a
; nonnegative bound for a defined external value.
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
define void @signed_guard(i32 noundef %face) "tensix-executor"="trisc0" {
entry:
  %active = icmp slt i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  br label %exit
exit:
  ret void
}

;--- non-dominating-guard.ll
; The bounded successor does not dominate the issuing block. Reusing its
; range at the merge would admit out-of-range values from the other edge.
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
declare void @llvm.riscv.tt.nop()
define void @non_dominating_guard(i32 noundef %face) "tensix-executor"="trisc0" {
entry:
  %active = icmp ult i32 %face, 4
  br i1 %active, label %bounded, label %merge
bounded:
  call void @llvm.riscv.tt.nop()
  br label %merge
merge:
  %right = and i32 %face, 1
  %lower = lshr i32 %face, 1
  %hi = shl i32 %right, 1
  %transposed = or i32 %hi, %lower
  %output.row = mul i32 %transposed, 16
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %output.row, i32 1, i32 1, i32 1)
  ret void
}

;--- unsafe-shift.ll
; Masking the result limits its numeric range but does not make an invalid
; shift count non-poison. The independent definedness obligation remains.
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
define void @unsafe_shift(i32 noundef %amount) "tensix-executor"="trisc0" {
  %count = and i32 %amount, 63
  %shifted = shl i32 1, %count
  %field = and i32 %shifted, 3
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %field, i32 1, i32 1, i32 1)
  ret void
}

;--- poison.ll
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
define void @poison_field() "tensix-executor"="trisc0" {
  %field = and i32 poison, 3
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %field, i32 1, i32 1, i32 1)
  ret void
}
