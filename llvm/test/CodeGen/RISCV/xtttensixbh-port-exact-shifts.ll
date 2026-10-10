; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/stream.ll -o - | FileCheck %s --check-prefix=STREAM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/stream.ll -o - | FileCheck %s --check-prefix=STREAM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/lshr.ll -o - | FileCheck %s --check-prefix=LSHR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/lshr.ll -o - | FileCheck %s --check-prefix=LSHR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/ashr.ll -o - | FileCheck %s --check-prefix=ASHR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/ashr.ll -o - | FileCheck %s --check-prefix=ASHR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/lost-bit.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/ashr-lost-bit.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/excess-count.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/assertion-only.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-count.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/undef-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-masked-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; InstCombine marks aligned stream-register address shifts exact. The flag
; needs an independent proof: the input's known zero bits must cover every
; possible shift count. That proof does not establish operand definedness,
; permit width-sized shifts, or repair earlier poison with a field mask.
; STREAM-LABEL: stream_address:
; STREAM: sw
; STREAM: ret
; LSHR-LABEL: lshr_aligned_loop:
; LSHR: sw
; LSHR: {{b(ne|lt|ge|eq)}}
; LSHR: ret
; ASHR-LABEL: ashr_aligned_loop:
; ASHR: sw
; ASHR: {{b(ne|lt|ge|eq)}}
; ASHR: ret
; POISON: STOREREG field regaddr must be defined and non-poison

;--- stream.ll
@runtime = external global ptr
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @stream_address() "tensix-executor"="trisc0" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base
  %bank = freeze i32 %word
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, 262176
  %regshift = lshr exact i32 %address, 2
  %regaddr = and i32 %regshift, 261128
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- lshr.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @lshr_aligned_loop() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %bank = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %bank, 16
  br i1 %active, label %body, label %exit
body:
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, 262176
  %count = and i32 %bank, 3
  %regshift = lshr exact i32 %address, %count
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  %next = add i32 %bank, 1
  br label %header
exit:
  ret void
}

;--- ashr.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @ashr_aligned_loop() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %bank = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %bank, 16
  br i1 %active, label %body, label %exit
body:
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, -2147483616
  %count = and i32 %bank, 3
  %regshift = ashr exact i32 %address, %count
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  %next = add i32 %bank, 1
  br label %header
exit:
  ret void
}

;--- lost-bit.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @lost_bit(i32 noundef %bank) "tensix-executor"="trisc0" {
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, 262178
  %regshift = lshr exact i32 %address, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- ashr-lost-bit.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @ashr_lost_bit(i32 noundef %bank) "tensix-executor"="trisc0" {
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, -2147483614
  %regshift = ashr exact i32 %address, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- excess-count.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @excess_count(i32 noundef %bank) "tensix-executor"="trisc0" {
  %count = and i32 %bank, 63
  %regshift = lshr exact i32 0, %count
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- assertion-only.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @assertion_only(i32 noundef %bank) "tensix-executor"="trisc0" {
  %bounded = and i32 %bank, 4095
  %regshift = lshr exact i32 %bounded, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- poison-value.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @poison_value() "tensix-executor"="trisc0" {
  %aligned = and i32 poison, -32
  %regshift = lshr exact i32 %aligned, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- poison-count.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @poison_count() "tensix-executor"="trisc0" {
  %count = and i32 poison, 3
  %regshift = lshr exact i32 262176, %count
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- undef-value.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @undef_value(i32 %bank) "tensix-executor"="trisc0" {
  %shifted = shl i32 %bank, 12
  %address = add i32 %shifted, 262176
  %regshift = lshr exact i32 %address, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}

;--- poison-masked-input.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @poison_masked_input(i32 noundef %bank) "tensix-executor"="trisc0" {
  %overflow = shl nuw i32 %bank, 31
  %aligned = and i32 %overflow, -32
  %regshift = lshr exact i32 %aligned, 2
  %regaddr = and i32 %regshift, 262143
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 12)
  ret void
}
