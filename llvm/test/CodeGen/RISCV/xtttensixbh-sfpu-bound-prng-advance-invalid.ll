; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/destination.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEST
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/ordinary-move.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; TIE: bound SFPU destructive tie requires identical physical registers
; DEST: writable LReg
; OLD: bound SFPU invalid read register operand class
; EXECUTOR: bound SFPU execution requires a TRISC executor
; MODE: bound SFPU unsupported mode at operand 3
;
; A dedicated sample action does not broaden ordinary SFPMOV or let CReg9
; masquerade as the stateful FROM_SPECIAL source. The sample ABI has no source
; selector and cannot request a preservation copy for mismatched old/dst.

;--- tie.ll
declare void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 immarg, i32 immarg)
define void @mismatched_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 1)
  ret void
}

;--- destination.ll
declare void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 immarg, i32 immarg)
define void @constant_destination() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 9, i32 9)
  ret void
}

;--- old.ll
declare void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 immarg, i32 immarg)
define void @constant_old_is_not_prng() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 9)
  ret void
}

;--- executor.ll
declare void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 immarg, i32 immarg)
define void @wrong_executor() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 0)
  ret void
}

;--- ordinary-move.ll
declare void @llvm.riscv.tt.bound.sfpmov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @ordinary_move_cannot_read_prng() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 9, i32 8)
  ret void
}
