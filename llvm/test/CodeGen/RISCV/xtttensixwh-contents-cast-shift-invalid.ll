; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/cast2.ll -o %t/cast2.o 2>&1 | FileCheck %s --check-prefix=CAST
; RUN: test ! -e %t/cast2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/cast3.ll -o %t/cast3.o 2>&1 | FileCheck %s --check-prefix=CAST
; RUN: test ! -e %t/cast3.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/vector2.ll -o %t/vector2.o 2>&1 | FileCheck %s --check-prefix=SHIFT
; RUN: test ! -e %t/vector2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/immediate3.ll -o %t/immediate3.o 2>&1 | FileCheck %s --check-prefix=SHIFT
; RUN: test ! -e %t/immediate3.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/immediate5.ll -o %t/immediate5.o 2>&1 | FileCheck %s --check-prefix=SHIFT
; RUN: test ! -e %t/immediate5.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/immediate7.ll -o %t/immediate7.o 2>&1 | FileCheck %s --check-prefix=SHIFT
; RUN: test ! -e %t/immediate7.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/too-negative.ll -o %t/too-negative.o 2>&1 | FileCheck %s --check-prefix=SIGNED
; RUN: test ! -e %t/too-negative.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/too-positive.ll -o %t/too-positive.o 2>&1 | FileCheck %s --check-prefix=SIGNED
; RUN: test ! -e %t/too-positive.o

; CAST: Wormhole SFPCAST operand 3 must be in [0, 1]
; SHIFT: Wormhole SFPSHFT operand 4 must be in [0, 1]
; SIGNED: Tensix immediate operand must fit in 12 signed bits

;--- cast2.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
define void @cast2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %old, <32 x i32> %old, i32 2)
  ret void
}

;--- cast3.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
define void @cast3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %old, <32 x i32> %old, i32 3)
  ret void
}

;--- vector2.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32>, <32 x i32>, i32 immarg)
define void @vector2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %old, <32 x i32> %old, i32 2)
  ret void
}

;--- immediate3.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @immediate3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %old, i32 -1, i32 3)
  ret void
}

;--- immediate5.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @immediate5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %old, i32 -1, i32 5)
  ret void
}

;--- immediate7.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @immediate7() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %old, i32 -1, i32 7)
  ret void
}

;--- too-negative.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @too_negative() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %old, i32 -2049, i32 1)
  ret void
}

;--- too-positive.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @too_positive() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %old, i32 2048, i32 1)
  ret void
}
