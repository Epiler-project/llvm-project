; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -enable-new-pm %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: unsupported Tensix intrinsic ABI: llvm.riscv.tt.bound.sfpnop
;
; This sole retired physical ABI fixture proves rejection, never an alias or
; fallback external call. Production and all other physical tests use new IDs.
declare void @llvm.riscv.tt.bound.sfpnop()
define void @retired() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpnop()
  ret void
}
