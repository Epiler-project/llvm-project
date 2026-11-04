; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/bh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/odd.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/index.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/no-reset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/no-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/predicated.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ACTIVE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESET
; BH: SFPLOADMACRO requires actual zero Dst address setup on every incoming path
; OFFSET: SFPLOADMACRO FP32 offset must be an even constant in [0, 1022]
; INDEX: SFPLOADMACRO macro index must be in [0, 3]
; RESET: SFPLOADMACRO requires actual zero Dst address setup on every incoming path
; CONFIG: SFPLOADMACRO requires its actual ADDI template, sequence and FP32 misc configuration
; ACTIVE: SFPLOADMACRO configuration and launch require all-active zero LaneConfig

;--- bh.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @bh() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- odd.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @odd() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 17, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- index.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @index() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- no-reset.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @no_reset() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- no-config.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @no_config() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- predicated.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @predicated() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  ; Enable predication with false lane flags; Enable=0 would make all active.
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}

;--- join.ll
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @conditional_reset(i1 %condition) "tensix-executor"="trisc1" {
entry:
  br i1 %condition, label %reset, label %join
reset:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  br label %join
join:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %next = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %old, i32 16, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 2, i32 0, i32 3)
  ret void
}
