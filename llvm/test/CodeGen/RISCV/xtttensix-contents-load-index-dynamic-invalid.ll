; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/unknown.ll -o %t/bh-unknown.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/bh-unknown.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/overflow.ll -o %t/bh-overflow.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/bh-overflow.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/poison.ll -o %t/bh-poison.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/bh-poison.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/bh-undef.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/bh-undef.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/unknown.ll -o %t/wh-unknown.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/wh-unknown.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/overflow.ll -o %t/wh-overflow.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/wh-overflow.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/poison.ll -o %t/wh-poison.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/wh-poison.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/wh-undef.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/wh-undef.o

; RANGE: contents paired LOAD Dst offset must be proven in [0, 1023]
; POISON: contents paired LOAD Dst offset must be defined and non-poison

;--- unknown.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)

define void @unknown(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %seed, <32 x i32> %seed, i32 %offset, i32 0, i32 3)
  ret void
}

;--- overflow.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)

define void @overflow(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bad = or i32 %offset, 1024
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %seed, <32 x i32> %seed, i32 %bad, i32 0, i32 3)
  ret void
}

;--- poison.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)

define void @poison(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %seed, <32 x i32> %seed, i32 poison, i32 0, i32 3)
  ret void
}

;--- undef.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)

define void @undef(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %seed, <32 x i32> %seed, i32 undef, i32 0, i32 3)
  ret void
}
