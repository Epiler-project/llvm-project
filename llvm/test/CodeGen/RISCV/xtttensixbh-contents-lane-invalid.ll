; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wide-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE_LOAD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/bad-unary-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD_UNARY_MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wide-exponent.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE_EXPONENT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wide-sign.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE_SIGN
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/immediate-form.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE_FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vector-form.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR_FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/signed-width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SIGNED_WIDTH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vector-shift-form.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR_SHIFT_FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/float-descale.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLOAT_DESCALE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/integer-descale.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INTEGER_DESCALE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/round-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROUND_MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/old-live.ll -o %t/old-live.o 2>&1 | FileCheck %s --check-prefix=OLD_LIVE
; RUN: test ! -e %t/old-live.o

; WIDE_LOAD: Tensix immediate operand must fit in 16 unsigned bits
; BAD_UNARY_MODE: SFPEXEXP mode must be one of {0, 1, 2, 3, 8, 9, 10, 11}
; WIDE_EXPONENT: Tensix field immediate must be in [0, 255]
; WIDE_SIGN: Tensix field immediate must be in [0, 1]
; IMMEDIATE_FORM: unsupported contents instruction form or mode
; VECTOR_FORM: unsupported contents instruction form or mode
; SIGNED_WIDTH: Tensix immediate operand must fit in 12 signed bits
; VECTOR_SHIFT_FORM: unsupported contents instruction form or mode
; FLOAT_DESCALE: Tensix floating conversion requires zero descale; integer descale must fit five bits
; INTEGER_DESCALE: Tensix floating conversion requires zero descale; integer descale must fit five bits
; ROUND_MODE: Tensix stochastic rounding mode must be in [0, 2]
; OLD_LIVE: Tensix SFPU cannot synthesize a register copy

;--- wide-load.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @invalid_wide_load() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65536, i32 2)
  ret void
}

;--- bad-unary-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_bad_unary_mode() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %old, <32 x i32> %old, i32 4)
  ret void
}

;--- wide-exponent.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @invalid_wide_exponent() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %old, i32 256, i32 1)
  ret void
}

;--- wide-sign.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @invalid_wide_sign() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %old, i32 2, i32 1)
  ret void
}

;--- immediate-form.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @invalid_immediate_form() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %old, i32 0, i32 0)
  ret void
}

;--- vector-form.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_vector_form() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %old, i32 1)
  ret void
}

;--- signed-width.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @invalid_signed_width() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %old, i32 -2049, i32 5)
  ret void
}

;--- vector-shift-form.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_vector_shift_form() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %old, <32 x i32> %old, i32 1)
  ret void
}

;--- float-descale.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @invalid_float_descale() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32> %old, <32 x i32> %old, i32 1, i32 0, i32 0)
  ret void
}

;--- integer-descale.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @invalid_integer_descale() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32> %old, <32 x i32> %old, i32 32, i32 4, i32 0)
  ret void
}

;--- round-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @invalid_round_mode() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32> %old, <32 x i32> %old, <32 x i32> %old, i32 4, i32 3)
  ret void
}

;--- old-live.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @old_destination_remains_live() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 7, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 4, i32 7, i32 3)
  ret void
}
