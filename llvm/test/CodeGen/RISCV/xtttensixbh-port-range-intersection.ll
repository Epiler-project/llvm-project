; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 %t/nsw-overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 %t/nuw-overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 %t/assertion-only.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 %t/poison-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 %t/undef-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; SCEV models unsigned remainder as a modular expression. Its unsigned range
; is [0,21), while its independently computed signed range can be full-set.
; Both ranges constrain the same operand bits. Their intersection proves that
; shifting by 12 satisfies both no-wrap flags; the flags themselves prove
; nothing. An input with a bounded result still needs its own definedness proof.
; CHECK-LABEL: partial_ring_address:
; CHECK: sw
; CHECK: ret
; CHECK-LABEL: loop_ring_address:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; POISON: SETDMAREG field payload_sigsel must be defined and non-poison

;--- valid.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)

define void @partial_ring_address(i32 noundef %sequence, i32 noundef %base) "tensix-executor"="trisc0" {
  %slot = urem i32 %sequence, 21
  %bytes = shl nuw nsw i32 %slot, 12
  %address = add i32 %base, %bytes
  %word = lshr i32 %address, 4
  %minus_one = add nsw i32 %word, -1
  %low = and i32 %minus_one, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}

define void @loop_ring_address(i32 noundef %start, i32 noundef %base) "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %iteration = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %iteration, 4
  br i1 %active, label %body, label %exit
body:
  %sequence = add i32 %start, %iteration
  %slot = urem i32 %sequence, 21
  %bytes = shl nuw nsw i32 %slot, 12
  %address = add i32 %base, %bytes
  %word = lshr i32 %address, 4
  %minus_one = add nsw i32 %word, -1
  %low = and i32 %minus_one, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  %next = add i32 %iteration, 1
  br label %header
exit:
  ret void
}

;--- nsw-overflow.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)
define void @nsw_overflow(i32 noundef %sequence) "tensix-executor"="trisc0" {
  %slot = urem i32 %sequence, 21
  ; 20 * 2^27 fits unsigned but exceeds the signed maximum.
  %bytes = shl nuw nsw i32 %slot, 27
  %word = lshr i32 %bytes, 4
  %low = and i32 %word, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}

;--- nuw-overflow.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)
define void @nuw_overflow(i32 noundef %sequence) "tensix-executor"="trisc0" {
  %slot = urem i32 %sequence, 21
  ; 20 * 2^28 exceeds the unsigned maximum independently of signedness.
  %bytes = shl nuw i32 %slot, 28
  %word = lshr i32 %bytes, 4
  %low = and i32 %word, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}

;--- assertion-only.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)
define void @assertion_only(i32 noundef %sequence) "tensix-executor"="trisc0" {
  %bytes = shl nuw nsw i32 %sequence, 12
  %word = lshr i32 %bytes, 4
  %low = and i32 %word, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}

;--- poison-input.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)
define void @poison_input() "tensix-executor"="trisc0" {
  %slot = urem i32 poison, 21
  %bytes = shl nuw nsw i32 %slot, 12
  %word = lshr i32 %bytes, 4
  %low = and i32 %word, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}

;--- undef-input.ll
declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)
define void @undef_input(i32 %sequence) "tensix-executor"="trisc0" {
  %slot = urem i32 %sequence, 21
  %bytes = shl nuw nsw i32 %slot, 12
  %word = lshr i32 %bytes, 4
  %low = and i32 %word, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 %low, i32 0)
  ret void
}
