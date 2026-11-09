; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/unknown.ll -o %t/bh-unknown.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/bh-unknown.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/outofrange.ll -o %t/bh-outofrange.o 2>&1 | FileCheck %s --check-prefix=OUTSIDE
; RUN: test ! -e %t/bh-outofrange.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/negative.ll -o %t/bh-negative.o 2>&1 | FileCheck %s --check-prefix=OUTSIDE
; RUN: test ! -e %t/bh-negative.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/poison.ll -o %t/bh-poison.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/bh-poison.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/bh-undef.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/bh-undef.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/unknown.ll -o %t/wh-unknown.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/wh-unknown.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/outofrange.ll -o %t/wh-outofrange.o 2>&1 | FileCheck %s --check-prefix=OUTSIDE
; RUN: test ! -e %t/wh-outofrange.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/negative.ll -o %t/wh-negative.o 2>&1 | FileCheck %s --check-prefix=OUTSIDE
; RUN: test ! -e %t/wh-negative.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/poison.ll -o %t/wh-poison.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/wh-poison.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/wh-undef.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/wh-undef.o

; RANGE: contents selected immediate selector must be proven within its finite group
; OUTSIDE: contents selected immediate selector exceeds its finite group
; POISON: contents selected immediate selector must be defined and non-poison

;--- unknown.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
define void @unknown(i32 noundef %selector) "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %b = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %a, <32 x i32> %b, i32 %selector, i32 16256, i32 8)
  ret void
}

;--- outofrange.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
define void @outofrange(i32 noundef %selector) "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %b = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %a, <32 x i32> %b, i32 2, i32 16256, i32 8)
  ret void
}

;--- negative.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
define void @negative(i32 noundef %selector) "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %b = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %a, <32 x i32> %b, i32 -1, i32 16256, i32 8)
  ret void
}

;--- poison.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
define void @poison(i32 noundef %selector) "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %b = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %a, <32 x i32> %b, i32 poison, i32 16256, i32 8)
  ret void
}

;--- undef.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
define void @undef(i32 noundef %selector) "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %b = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %a, <32 x i32> %b, i32 undef, i32 16256, i32 8)
  ret void
}
