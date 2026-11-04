; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/div-range.ll -o %t/div-range.o 2>&1 | FileCheck %s --check-prefix=E0
; RUN: test ! -e %t/div-range.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/div-mode.ll -o %t/div-mode.o 2>&1 | FileCheck %s --check-prefix=E1
; RUN: test ! -e %t/div-mode.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/fp-range.ll -o %t/fp-range.o 2>&1 | FileCheck %s --check-prefix=E2
; RUN: test ! -e %t/fp-range.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/fp-reserved.ll -o %t/fp-reserved.o 2>&1 | FileCheck %s --check-prefix=E3
; RUN: test ! -e %t/fp-reserved.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/fp-indirect.ll -o %t/fp-indirect.o 2>&1 | FileCheck %s --check-prefix=E4
; RUN: test ! -e %t/fp-indirect.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/wh-add-negate.ll -o %t/wh-add-negate.o 2>&1 | FileCheck %s --check-prefix=E5
; RUN: test ! -e %t/wh-add-negate.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/wh-mul-negate.ll -o %t/wh-mul-negate.o 2>&1 | FileCheck %s --check-prefix=E6
; RUN: test ! -e %t/wh-mul-negate.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/wh-mul23.ll -o %t/wh-mul23.o 2>&1 | FileCheck %s --check-prefix=E7
; RUN: test ! -e %t/wh-mul23.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/mul23-indirect.ll -o %t/mul23-indirect.o 2>&1 | FileCheck %s --check-prefix=E8
; RUN: test ! -e %t/mul23-indirect.o

; E0: Tensix immediate operand must fit in 8 unsigned bits
;--- div-range.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %src, i32 256, i32 0)
  ret void
}

; E1: Tensix immediate operand must fit in 1 unsigned bits
;--- div-mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %src, i32 1, i32 2)
  ret void
}

; E2: Tensix immediate operand must fit in 16 unsigned bits
;--- fp-range.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %old, i32 65536, i32 0)
  ret void
}

; E3: SFPADDI mode must be one of {0, 2}
;--- fp-reserved.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %old, i32 16256, i32 1)
  ret void
}

; E4: SFPMULI mode must be one of {0, 2}
;--- fp-indirect.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_4() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %old, i32 16384, i32 8)
  ret void
}

; E5: Wormhole SFPADDI operand 3 must be in [0, 0]
;--- wh-add-negate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %old, i32 16256, i32 2)
  ret void
}

; E6: Wormhole SFPMULI operand 3 must be in [0, 0]
;--- wh-mul-negate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %old, i32 16384, i32 2)
  ret void
}

; E7: Tensix instruction is not supported on Wormhole
;--- wh-mul23.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_7() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32> %old, <32 x i32> %src, <32 x i32> %src, i32 0)
  ret void
}

; E8: Tensix immediate operand must fit in 1 unsigned bits
;--- mul23-indirect.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32> %old, <32 x i32> %src, <32 x i32> %src, i32 4)
  ret void
}
