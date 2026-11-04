; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/exexp.ll -o %t/exexp.o 2>&1 | FileCheck %s --check-prefix=EXEXP
; RUN: test ! -e %t/exexp.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/exman.ll -o %t/exman.o 2>&1 | FileCheck %s --check-prefix=EXMAN
; RUN: test ! -e %t/exman.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/lz.ll -o %t/lz.o 2>&1 | FileCheck %s --check-prefix=LZ
; RUN: test ! -e %t/lz.o

; These reserved mode bits must remain rejected when the three operations
; are enabled on WH. Current silicon's behavior for reserved modifiers does
; not authorize a contents operation.
; EXEXP: SFPEXEXP mode must be one of {0, 1, 2, 3, 8, 9, 10, 11}
; EXMAN: SFPEXMAN mode must be one of {0, 1}
; LZ: SFPLZ mode must be one of {0, 2, 4, 6, 8, 10, 12, 14}

;--- exexp.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32>, <32 x i32>, i32 immarg)
define void @reserved_exexp_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %old, <32 x i32> %old, i32 4)
  ret void
}

;--- exman.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32>, <32 x i32>, i32 immarg)
define void @reserved_exman_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %old, <32 x i32> %old, i32 2)
  ret void
}

;--- lz.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
define void @reserved_lz_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %old, <32 x i32> %old, i32 1)
  ret void
}
