; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-bh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-bh-o2.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-wh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-wh-o2.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-wh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/masked.ll -o %t/masked-bh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/masked.ll -o %t/masked-bh-o2.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/masked.ll -o %t/masked-wh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/masked.ll -o %t/masked-wh-o2.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-wh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/half-upper.ll -o %t/half-upper-bh-o0.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-upper-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/half-upper.ll -o %t/half-upper-bh-o2.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-upper-bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/half-upper.ll -o %t/half-upper-wh-o0.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-upper-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/half-upper.ll -o %t/half-upper-wh-o2.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-upper-wh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/half-lower.ll -o %t/half-lower-bh-o0.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-lower-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/half-lower.ll -o %t/half-lower-bh-o2.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-lower-bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/half-lower.ll -o %t/half-lower-wh-o0.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-lower-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/half-lower.ll -o %t/half-lower-wh-o2.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not test -e %t/half-lower-wh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/record.ll -o %t/record-bh-o0.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/record-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/record.ll -o %t/record-bh-o2.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/record-bh-o2.o

; STATE: contents full immediate requires all lanes writable with unmasked rows
; MODE: contents full immediate requires a complete-word mode
; RECORD: contents full immediate is not supported in replay recording

;--- no-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
define void @reject_no_reset() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 0)
  ret void
}

;--- masked.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
define void @reject_masked() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 0)
  ret void
}

;--- half-upper.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
define void @reject_half_upper() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 8)
  ret void
}

;--- half-lower.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
define void @reject_half_lower() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 10)
  ret void
}

;--- record.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
define void @reject_record() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 1, i32 0)
  ret void
}
