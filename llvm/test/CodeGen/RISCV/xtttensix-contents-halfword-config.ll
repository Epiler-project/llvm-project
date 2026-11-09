; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs -filetype=obj %t/all-lanes.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/inactive-lanes.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/masked-rows.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/index-enabled.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/upper-only.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/all-lanes.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/inactive-lanes.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/masked-rows.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/index-enabled.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/upper-only.ll -o /dev/null 2>&1 | FileCheck %s

; CHECK: contents SFPU requires ENABLE_DEST_INDEX known disabled
; Two authored halfword writes may establish index-bit zero from unknown old
; contents only with both full effective CC and unmasked rows. The negative
; cases retain unknown inactive bits or explicitly set the index bit.

;--- all-lanes.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @halfword_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %hi = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 0, i32 8)
  %lo = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %hi, i32 61440, i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %lo, i32 21844, i32 8)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %lo, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 0, i32 0, i32 2)
  ret void
}

;--- inactive-lanes.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @halfword_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %hi = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 0, i32 8)
  %lo = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %hi, i32 61440, i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %lo, i32 21844, i32 8)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %lo, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 0, i32 0, i32 2)
  ret void
}

;--- masked-rows.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @halfword_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %zero, i32 61440, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %mask, i32 21844, i32 8)
  %hi = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 0, i32 8)
  %lo = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %hi, i32 61440, i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %lo, i32 21844, i32 8)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %lo, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 0, i32 0, i32 2)
  ret void
}

;--- index-enabled.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @halfword_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %hi = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 0, i32 8)
  %lo = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %hi, i32 61444, i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %lo, i32 21844, i32 8)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %lo, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 0, i32 0, i32 2)
  ret void
}

;--- upper-only.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @halfword_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %hi = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 0, i32 8)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %hi, i32 21844, i32 8)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %hi, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 0, i32 0, i32 2)
  ret void
}
