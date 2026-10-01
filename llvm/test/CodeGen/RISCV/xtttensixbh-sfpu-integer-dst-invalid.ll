; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie5.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie6.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie8.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset5.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset6.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset8.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison5.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison6.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison8.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; TIE: tie
; OFFSET: Tensix immediate operand must fit in 10 unsigned bits
; POISON: bound Dst offset must be defined and non-poison
;--- tie5.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @tie5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 1, i32 0, i32 7, i32 5)
  ret void
}
;--- tie6.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @tie6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 1, i32 0, i32 7, i32 6)
  ret void
}
;--- tie8.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @tie8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 1, i32 0, i32 7, i32 8)
  ret void
}
;--- offset5.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @offset5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 1024, i32 7, i32 5)
  ret void
}
;--- offset6.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @offset6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 1024, i32 7, i32 6)
  ret void
}
;--- offset8.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @offset8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 1024, i32 7, i32 8)
  ret void
}
;--- poison5.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @poison5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 poison, i32 7, i32 5)
  ret void
}
;--- poison6.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @poison6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 poison, i32 7, i32 6)
  ret void
}
;--- poison8.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @poison8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 poison, i32 7, i32 8)
  ret void
}
