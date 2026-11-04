; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_wide.ll -o %t/setexp_wide-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_WIDE
; RUN: test ! -e %t/setexp_wide-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_wide.ll -o %t/setexp_wide-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_WIDE
; RUN: test ! -e %t/setexp_wide-bh.o
; SETEXP_WIDE: Tensix field immediate must be in [0, 255]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_negative.ll -o %t/setexp_negative-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_NEGATIVE
; RUN: test ! -e %t/setexp_negative-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_negative.ll -o %t/setexp_negative-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_NEGATIVE
; RUN: test ! -e %t/setexp_negative-bh.o
; SETEXP_NEGATIVE: Tensix immediate operand must fit in 16 unsigned bits
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_mode.ll -o %t/setexp_mode-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_MODE
; RUN: test ! -e %t/setexp_mode-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_mode.ll -o %t/setexp_mode-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_MODE
; RUN: test ! -e %t/setexp_mode-bh.o
; SETEXP_MODE: SFPSETEXP mode must be one of
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_immediate-form.ll -o %t/setexp_immediate-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_IMMEDIATE_FORM
; RUN: test ! -e %t/setexp_immediate-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_immediate-form.ll -o %t/setexp_immediate-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_IMMEDIATE_FORM
; RUN: test ! -e %t/setexp_immediate-form-bh.o
; SETEXP_IMMEDIATE_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_vector-form.ll -o %t/setexp_vector-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_VECTOR_FORM
; RUN: test ! -e %t/setexp_vector-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_vector-form.ll -o %t/setexp_vector-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_VECTOR_FORM
; RUN: test ! -e %t/setexp_vector-form-bh.o
; SETEXP_VECTOR_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_old-live-immediate.ll -o %t/setexp_old-live-immediate-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setexp_old-live-immediate-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_old-live-immediate.ll -o %t/setexp_old-live-immediate-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setexp_old-live-immediate-bh.o
; SETEXP_OLD_LIVE_IMMEDIATE: Tensix SFPU cannot synthesize a register copy
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setexp_old-live-vector.ll -o %t/setexp_old-live-vector-wh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setexp_old-live-vector-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setexp_old-live-vector.ll -o %t/setexp_old-live-vector-bh.o 2>&1 | FileCheck %s --check-prefix=SETEXP_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setexp_old-live-vector-bh.o
; SETEXP_OLD_LIVE_VECTOR: Tensix SFPU cannot synthesize a register copy
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_wide.ll -o %t/setman_wide-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_WIDE
; RUN: test ! -e %t/setman_wide-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_wide.ll -o %t/setman_wide-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_WIDE
; RUN: test ! -e %t/setman_wide-bh.o
; SETMAN_WIDE: Tensix field immediate must be in [0, 4095]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_negative.ll -o %t/setman_negative-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_NEGATIVE
; RUN: test ! -e %t/setman_negative-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_negative.ll -o %t/setman_negative-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_NEGATIVE
; RUN: test ! -e %t/setman_negative-bh.o
; SETMAN_NEGATIVE: Tensix immediate operand must fit in 16 unsigned bits
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_mode.ll -o %t/setman_mode-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_MODE
; RUN: test ! -e %t/setman_mode-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_mode.ll -o %t/setman_mode-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_MODE
; RUN: test ! -e %t/setman_mode-bh.o
; SETMAN_MODE: SFPSETMAN mode must be one of
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_immediate-form.ll -o %t/setman_immediate-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_IMMEDIATE_FORM
; RUN: test ! -e %t/setman_immediate-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_immediate-form.ll -o %t/setman_immediate-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_IMMEDIATE_FORM
; RUN: test ! -e %t/setman_immediate-form-bh.o
; SETMAN_IMMEDIATE_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_vector-form.ll -o %t/setman_vector-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_VECTOR_FORM
; RUN: test ! -e %t/setman_vector-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_vector-form.ll -o %t/setman_vector-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_VECTOR_FORM
; RUN: test ! -e %t/setman_vector-form-bh.o
; SETMAN_VECTOR_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_old-live-immediate.ll -o %t/setman_old-live-immediate-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setman_old-live-immediate-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_old-live-immediate.ll -o %t/setman_old-live-immediate-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setman_old-live-immediate-bh.o
; SETMAN_OLD_LIVE_IMMEDIATE: Tensix SFPU cannot synthesize a register copy
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setman_old-live-vector.ll -o %t/setman_old-live-vector-wh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setman_old-live-vector-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setman_old-live-vector.ll -o %t/setman_old-live-vector-bh.o 2>&1 | FileCheck %s --check-prefix=SETMAN_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setman_old-live-vector-bh.o
; SETMAN_OLD_LIVE_VECTOR: Tensix SFPU cannot synthesize a register copy
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_wide.ll -o %t/setsgn_wide-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_WIDE
; RUN: test ! -e %t/setsgn_wide-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_wide.ll -o %t/setsgn_wide-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_WIDE
; RUN: test ! -e %t/setsgn_wide-bh.o
; SETSGN_WIDE: Tensix field immediate must be in [0, 1]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_negative.ll -o %t/setsgn_negative-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_NEGATIVE
; RUN: test ! -e %t/setsgn_negative-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_negative.ll -o %t/setsgn_negative-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_NEGATIVE
; RUN: test ! -e %t/setsgn_negative-bh.o
; SETSGN_NEGATIVE: Tensix immediate operand must fit in 16 unsigned bits
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_mode.ll -o %t/setsgn_mode-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_MODE
; RUN: test ! -e %t/setsgn_mode-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_mode.ll -o %t/setsgn_mode-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_MODE
; RUN: test ! -e %t/setsgn_mode-bh.o
; SETSGN_MODE: SFPSETSGN mode must be one of
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_immediate-form.ll -o %t/setsgn_immediate-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_IMMEDIATE_FORM
; RUN: test ! -e %t/setsgn_immediate-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_immediate-form.ll -o %t/setsgn_immediate-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_IMMEDIATE_FORM
; RUN: test ! -e %t/setsgn_immediate-form-bh.o
; SETSGN_IMMEDIATE_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_vector-form.ll -o %t/setsgn_vector-form-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_VECTOR_FORM
; RUN: test ! -e %t/setsgn_vector-form-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_vector-form.ll -o %t/setsgn_vector-form-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_VECTOR_FORM
; RUN: test ! -e %t/setsgn_vector-form-bh.o
; SETSGN_VECTOR_FORM: unsupported contents instruction form or mode
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_old-live-immediate.ll -o %t/setsgn_old-live-immediate-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setsgn_old-live-immediate-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_old-live-immediate.ll -o %t/setsgn_old-live-immediate-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_OLD_LIVE_IMMEDIATE
; RUN: test ! -e %t/setsgn_old-live-immediate-bh.o
; SETSGN_OLD_LIVE_IMMEDIATE: Tensix SFPU cannot synthesize a register copy
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_old-live-vector.ll -o %t/setsgn_old-live-vector-wh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setsgn_old-live-vector-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/setsgn_old-live-vector.ll -o %t/setsgn_old-live-vector-bh.o 2>&1 | FileCheck %s --check-prefix=SETSGN_OLD_LIVE_VECTOR
; RUN: test ! -e %t/setsgn_old-live-vector-bh.o
; SETSGN_OLD_LIVE_VECTOR: Tensix SFPU cannot synthesize a register copy
;--- setexp_wide.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_wide() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 256, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_negative.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_negative() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 -1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_mode() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_immediate-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_immediate_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_vector-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_vector_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_old-live-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_old_live_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 255, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setexp_old-live-vector.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setexp_old_live_vector() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_wide.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_wide() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 4096, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_negative.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_negative() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 -1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_mode() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32> %old, <32 x i32> %source, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_immediate-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_immediate_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_vector-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_vector_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32> %old, <32 x i32> %source, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_old-live-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_old_live_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 4095, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setman_old-live-vector.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setman_old_live_vector() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_wide.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_wide() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 2, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_negative.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_negative() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 -1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_mode() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %old, <32 x i32> %source, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_immediate-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_immediate_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_vector-form.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_vector_form() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %old, <32 x i32> %source, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_old-live-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_old_live_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

;--- setsgn_old-live-vector.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setsgn_old_live_vector() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 1, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}
