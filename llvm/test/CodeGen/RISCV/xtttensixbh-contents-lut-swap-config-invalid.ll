; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/swap-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E0
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/lut-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E1
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/six-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E2
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/three-sign.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E3
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/creg-destination.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E4
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/creg-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E5
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/lane-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E6
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/mask-overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E7
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/nonzero-unmasked.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E8
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/odd-mask.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E9
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/lane-reaching.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REACHING
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/lut-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E1
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/six-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E2
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/three-sign.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=E3

; E0: contents swap mode must be in 0..9
; E1: contents LUT mode must be 0 or 4
; E2: contents LUT six mode must be 0,2,3,4,6 or 7
; E3: contents LUT three sign must be 0 or 1
; E4: contents configuration requires C11..C14
; E5: contents CReg configuration mode must be 0 or 8
; E6: contents lane configuration mode must be 8
; E7: contents configuration mask must fit u16
; E8: SFPCONFIG mode 0 requires a zero mask
; E9: SFPCONFIG mode 8 mask must select even lane bits
; REACHING: contents SFPU requires ENABLE_DEST_INDEX known disabled

;--- swap-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %r = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %v, <32 x i32> %v, i32 10)
  ret void
}

;--- lut-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 8)
  ret void
}

;--- six-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lutfp32.six(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.contents.lutfp32.six(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 10)
  ret void
}

;--- three-sign.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.lutfp32.three(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %r = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.lutfp32.three(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 2)
  ret void
}

;--- creg-destination.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %v, i32 10, i32 0, i32 0)
  ret void
}

;--- creg-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %v, i32 11, i32 0, i32 1)
  ret void
}

;--- lane-mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %v, i32 0, i32 0)
  ret void
}

;--- mask-overflow.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
define void @invalid() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %v, i32 65536, i32 8)
  ret void
}

;--- lane-reaching.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
define void @unknown_lane_configuration() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %source = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %source, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %unsupported = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  ret void
}

;--- nonzero-unmasked.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @nonzero_unmasked() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %v, i32 11, i32 1, i32 0)
  ret void
}

;--- odd-mask.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
define void @odd_mask() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %v, i32 2, i32 8)
  ret void
}
