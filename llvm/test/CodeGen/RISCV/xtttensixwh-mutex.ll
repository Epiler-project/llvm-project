; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/reserved.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESERVED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/extent.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXTENT
; CHECK: ttatgetm{{.*}}7
; CHECK: ttatrelm{{.*}}7
; RESERVED: Wormhole ATGETM does not match an admitted logical form
; EXTENT: Wormhole ATRELM field mutex_index must be in [0, 7]

;--- valid.ll
declare void @llvm.tt.atgetm(i32 immarg)
declare void @llvm.tt.atrelm(i32 immarg)
define void @valid() "tensix-executor"="trisc2" {
  call void @llvm.tt.atgetm(i32 7)
  call void @llvm.tt.atrelm(i32 7)
  ret void
}

;--- reserved.ll
declare void @llvm.tt.atgetm(i32 immarg)
define void @reserved() "tensix-executor"="trisc1" {
  call void @llvm.tt.atgetm(i32 1)
  ret void
}

;--- extent.ll
declare void @llvm.tt.atrelm(i32 immarg)
define void @extent() "tensix-executor"="trisc1" {
  call void @llvm.tt.atrelm(i32 8)
  ret void
}
