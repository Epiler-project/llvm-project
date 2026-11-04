; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/pressure8.ll > %t/pressure8.input.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/pressure8.input.ll -o %t/pressure8.bh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/pressure8.input.ll -o %t/pressure8.bh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/pressure8.input.ll -o %t/pressure8.wh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/pressure8.input.ll -o %t/pressure8.wh.O2.o
; RUN: cat %t/common.ll %t/pressure9.ll > %t/pressure9.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/pressure9.input.ll -o %t/pressure9.bh.O0.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: not test -e %t/pressure9.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/pressure9.input.ll -o %t/pressure9.bh.O2.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: not test -e %t/pressure9.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/pressure9.input.ll -o %t/pressure9.wh.O0.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: not test -e %t/pressure9.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/pressure9.input.ll -o %t/pressure9.wh.O2.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: not test -e %t/pressure9.wh.O2.o
; PRESSURE: ran out of registers during register allocation

;--- common.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)

;--- pressure8.ll
define void @pressure8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 61, i32 0, i32 0, i32 0, i32 1, <32 x i32> %v0, <32 x i32> %v1)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 61, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 61, <32 x i32> %v0)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 61, i32 1, <32 x i32> %v0, <32 x i32> %v1)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v4, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v5, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v6, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v7, i32 56, i32 0, i32 4)
  ret void
}

;--- pressure9.ll
define void @pressure9() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v8 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 61, i32 0, i32 0, i32 0, i32 1, <32 x i32> %v0, <32 x i32> %v1)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 61, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 61, <32 x i32> %v0)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 61, i32 1, <32 x i32> %v0, <32 x i32> %v1)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v4, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v5, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v6, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v7, i32 56, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v8, i32 64, i32 0, i32 4)
  ret void
}
