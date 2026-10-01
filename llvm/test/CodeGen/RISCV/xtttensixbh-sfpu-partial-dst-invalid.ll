; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie14.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tie15.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/stack10.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/integer13.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; TIE: tie
; MODE: format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
; OFFSET: bound Dst offset must be defined and non-poison
;--- tie14.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @tie14() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 1, i32 0, i32 7, i32 14)
  ret void
}
;--- tie15.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @tie15() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 1, i32 0, i32 7, i32 15)
  ret void
}
;--- stack10.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @stack10() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 7, i32 10)
  ret void
}
;--- integer13.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @integer13() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 7, i32 13)
  ret void
}
;--- offset.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @bad_offset(i10 %index) "tensix-executor"="trisc1" {
  %offset = zext i10 %index to i32
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %offset, i32 7, i32 14)
  ret void
}
