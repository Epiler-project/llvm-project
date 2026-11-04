; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/good.ll > %t/good-input.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/good-input.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/good-input.ll -o /dev/null
; RUN: cat %t/common.ll %t/bad-placement.ll > %t/bad-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/bad-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: cat %t/common.ll %t/bad-capacity.ll > %t/large-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/large-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAPACITY
; PLACEMENT: structured replay fixed placement conflicts or exceeds capacity
; CAPACITY: contents replay body exceeds replay capacity
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


;--- good.ll
define void @full_bank_32() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cap = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 1, i32 0, i32 1, <32 x i32> %old, <32 x i32> %cap)
  %x0 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %cap, i32 4)
  %x1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x0, <32 x i32> %cap, i32 4)
  %x2 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x1, <32 x i32> %cap, i32 4)
  %x3 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x2, <32 x i32> %cap, i32 4)
  %x4 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x3, <32 x i32> %cap, i32 4)
  %x5 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x4, <32 x i32> %cap, i32 4)
  %x6 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x5, <32 x i32> %cap, i32 4)
  %x7 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x6, <32 x i32> %cap, i32 4)
  %x8 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x7, <32 x i32> %cap, i32 4)
  %x9 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x8, <32 x i32> %cap, i32 4)
  %x10 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x9, <32 x i32> %cap, i32 4)
  %x11 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x10, <32 x i32> %cap, i32 4)
  %x12 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x11, <32 x i32> %cap, i32 4)
  %x13 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x12, <32 x i32> %cap, i32 4)
  %x14 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x13, <32 x i32> %cap, i32 4)
  %x15 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x14, <32 x i32> %cap, i32 4)
  %x16 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x15, <32 x i32> %cap, i32 4)
  %x17 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x16, <32 x i32> %cap, i32 4)
  %x18 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x17, <32 x i32> %cap, i32 4)
  %x19 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x18, <32 x i32> %cap, i32 4)
  %x20 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x19, <32 x i32> %cap, i32 4)
  %x21 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x20, <32 x i32> %cap, i32 4)
  %x22 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x21, <32 x i32> %cap, i32 4)
  %x23 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x22, <32 x i32> %cap, i32 4)
  %x24 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x23, <32 x i32> %cap, i32 4)
  %x25 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x24, <32 x i32> %cap, i32 4)
  %x26 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x25, <32 x i32> %cap, i32 4)
  %x27 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x26, <32 x i32> %cap, i32 4)
  %x28 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x27, <32 x i32> %cap, i32 4)
  %x29 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x28, <32 x i32> %cap, i32 4)
  %x30 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x29, <32 x i32> %cap, i32 4)
  %x31 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x30, <32 x i32> %cap, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %x31)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %x31, <32 x i32> %cap)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}
define void @last_slot_1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cap = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 1, i32 31, i32 1, <32 x i32> %old, <32 x i32> %cap)
  %x0 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %cap, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %x0)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %x0, <32 x i32> %cap)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

;--- bad-placement.ll
define void @range_crosses_bank() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cap = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 1, i32 31, i32 1, <32 x i32> %old, <32 x i32> %cap)
  %x0 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %cap, i32 4)
  %x1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x0, <32 x i32> %cap, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %x1)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %x1, <32 x i32> %cap)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

;--- bad-capacity.ll
define void @body_exceeds_bank() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cap = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 1, i32 0, i32 1, <32 x i32> %old, <32 x i32> %cap)
  %x0 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %cap, i32 4)
  %x1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x0, <32 x i32> %cap, i32 4)
  %x2 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x1, <32 x i32> %cap, i32 4)
  %x3 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x2, <32 x i32> %cap, i32 4)
  %x4 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x3, <32 x i32> %cap, i32 4)
  %x5 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x4, <32 x i32> %cap, i32 4)
  %x6 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x5, <32 x i32> %cap, i32 4)
  %x7 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x6, <32 x i32> %cap, i32 4)
  %x8 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x7, <32 x i32> %cap, i32 4)
  %x9 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x8, <32 x i32> %cap, i32 4)
  %x10 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x9, <32 x i32> %cap, i32 4)
  %x11 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x10, <32 x i32> %cap, i32 4)
  %x12 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x11, <32 x i32> %cap, i32 4)
  %x13 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x12, <32 x i32> %cap, i32 4)
  %x14 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x13, <32 x i32> %cap, i32 4)
  %x15 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x14, <32 x i32> %cap, i32 4)
  %x16 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x15, <32 x i32> %cap, i32 4)
  %x17 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x16, <32 x i32> %cap, i32 4)
  %x18 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x17, <32 x i32> %cap, i32 4)
  %x19 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x18, <32 x i32> %cap, i32 4)
  %x20 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x19, <32 x i32> %cap, i32 4)
  %x21 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x20, <32 x i32> %cap, i32 4)
  %x22 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x21, <32 x i32> %cap, i32 4)
  %x23 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x22, <32 x i32> %cap, i32 4)
  %x24 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x23, <32 x i32> %cap, i32 4)
  %x25 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x24, <32 x i32> %cap, i32 4)
  %x26 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x25, <32 x i32> %cap, i32 4)
  %x27 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x26, <32 x i32> %cap, i32 4)
  %x28 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x27, <32 x i32> %cap, i32 4)
  %x29 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x28, <32 x i32> %cap, i32 4)
  %x30 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x29, <32 x i32> %cap, i32 4)
  %x31 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x30, <32 x i32> %cap, i32 4)
  %x32 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %x31, <32 x i32> %cap, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %x32)
  %result.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %x32, <32 x i32> %cap)
  %result = extractvalue { <32 x i32> } %result.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}
