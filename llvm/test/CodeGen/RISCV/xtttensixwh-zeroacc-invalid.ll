; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/where.ll 2>&1 | FileCheck %s --check-prefix=WHERE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/addr.ll 2>&1 | FileCheck %s --check-prefix=ADDR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/flags.ll 2>&1 | FileCheck %s --check-prefix=FLAGS
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/use32.ll 2>&1 | FileCheck %s --check-prefix=USE32
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/half.ll 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/packed-mode.ll 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not opt -passes=verify -disable-output %t/dynamic.ll 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not opt -passes=verify -disable-output %t/undef.ll 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not opt -passes=verify -disable-output %t/poison.ll 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/port.ll 2>&1 | FileCheck %s --check-prefix=FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/mop.ll 2>&1 | FileCheck %s --check-prefix=FORM
; RUN: not llc -mtriple=riscv32 < %t/no-feature.ll 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/ncrisc.ll 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; WHERE: Wormhole ZEROACC field where must be in [0, 0]
; ADDR: Wormhole ZEROACC field addr_mode must be in [1, 1]
; FLAGS: Wormhole ZEROACC field clear_zero_flags must be in [0, 0]
; USE32: Wormhole ZEROACC field use_32_bit_mode must be in [0, 1]
; MODE: Wormhole ZEROACC field clear_mode must be in [3, 3]
; IMMEDIATE: immarg operand has non-immediate parameter
; FORM: Tensix intrinsic is not supported on Wormhole
; FEATURE: Tensix intrinsic requires +xtttensixbh
; EXECUTOR: ncrisc cannot issue Tensix instructions

;--- where.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 1, i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- addr.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 2, i32 0, i32 0, i32 3)
  ret void
}
;--- flags.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 1, i32 0, i32 3)
  ret void
}
;--- use32.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 2, i32 3)
  ret void
}
;--- half.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 0, i32 2)
  ret void
}
;--- packed-mode.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 1, i32 7)
  ret void
}
;--- dynamic.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad(i32 %where) "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 %where, i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- undef.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 undef, i32 3)
  ret void
}
;--- poison.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 poison, i32 3)
  ret void
}
;--- port.ll
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 0, i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- mop.ll
declare void @llvm.riscv.tt.zeroacc.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc.mop(i32 2, i32 0, i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- no-feature.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- ncrisc.ll
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 0, i32 3)
  ret void
}
