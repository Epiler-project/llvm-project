; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -global-isel -global-isel-abort=1 %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O0 %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O2 %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: Tensix SFPU requires the SelectionDAG legacy codegen pipeline

declare void @llvm.riscv.tt.sfpnop()
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpnop()
  ret void
}
