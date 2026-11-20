; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/acquire.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/release.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/dynamic-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMMEDIATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/mop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX

; INDEX: Blackhole mutex index must be 0, 2, 3, or 4
; IMMEDIATE: expected an immediate operand

;--- valid.ll
declare void @llvm.riscv.tt.atgetm(i32 immarg)
declare void @llvm.riscv.tt.atrelm(i32 immarg)
declare void @llvm.riscv.tt.atgetm.port(i32 immarg, i32)
declare void @llvm.riscv.tt.atrelm.port(i32 immarg, i32)
declare void @llvm.riscv.tt.atgetm.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.atrelm.mop(i32 immarg, i32 immarg)

define void @valid_indices() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.atgetm(i32 0)
  call void @llvm.riscv.tt.atrelm(i32 0)
  call void @llvm.riscv.tt.atgetm(i32 2)
  call void @llvm.riscv.tt.atrelm(i32 2)
  call void @llvm.riscv.tt.atgetm.port(i32 0, i32 3)
  call void @llvm.riscv.tt.atrelm.port(i32 0, i32 3)
  call void @llvm.riscv.tt.atgetm.mop(i32 2, i32 4)
  call void @llvm.riscv.tt.atrelm.mop(i32 3, i32 4)
  ret void
}

;--- acquire.ll
declare void @llvm.riscv.tt.atgetm(i32 immarg)
define void @bad_acquire() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.atgetm(i32 1)
  ret void
}

;--- release.ll
declare void @llvm.riscv.tt.atrelm(i32 immarg)
define void @bad_release() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.atrelm(i32 5)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.atrelm.port(i32 immarg, i32)
define void @bad_port() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.atrelm.port(i32 0, i32 1)
  ret void
}

;--- dynamic-port.ll
declare void @llvm.riscv.tt.atgetm.port(i32 immarg, i32)
define void @dynamic_port(i32 noundef %input) "tensix-executor"="trisc1" {
  %index = and i32 %input, 3
  call void @llvm.riscv.tt.atgetm.port(i32 0, i32 %index)
  ret void
}

;--- mop.ll
declare void @llvm.riscv.tt.atgetm.mop(i32 immarg, i32 immarg)
define void @bad_mop() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.atgetm.mop(i32 2, i32 5)
  ret void
}
