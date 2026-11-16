; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/preserve.ll -o %t/preserve-wh.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/preserve.ll -o %t/preserve-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/write-mode2-immediate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/write-mode2-immediate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/write-mode2-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOAD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/write-mode2-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOAD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/write-mode8-immediate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/write-mode8-immediate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/write-mode8-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOAD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/write-mode8-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOAD

; Modes 0/4 retain the actual enabled-lane proof across LZ. Full forms
; require that proof because they omit a pre-existing old destination.
; Both LZ and the subsequent full result stay live through target stores.
; Modes 2 (CC_NE0) and 8 (CC_COMP without CC_NE0) really write flags and
; must not inherit the preserving modes' proof, even with valid row setup.
; IMMEDIATE: contents full immediate requires all lanes writable with unmasked rows
; LOAD: contents full LOAD requires all lanes writable

;--- preserve.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @lz_mode0_then_full_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 0)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}
define void @lz_mode0_then_full_load() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 0)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}
define void @lz_mode4_then_full_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 4)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}
define void @lz_mode4_then_full_load() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 4)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}

;--- write-mode2-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @lz_mode2_then_full_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 2)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}

;--- write-mode2-load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @lz_mode2_then_full_load() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 2)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}

;--- write-mode8-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @lz_mode8_then_full_immediate() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 8)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}

;--- write-mode8-load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @lz_mode8_then_full_load() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %zero, i32 8)
  %after = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 4, i32 0, i32 3)
  ret void
}
