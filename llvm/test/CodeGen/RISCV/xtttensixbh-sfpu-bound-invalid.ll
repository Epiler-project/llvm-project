; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unbound.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT-UNBOUND
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/tie.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT-MISSING-COPY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/group.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT-IMPLICIT-GROUP
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/number.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NUMBER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; REJECT-UNBOUND: bound and legacy SFPU ingress cannot share a function
; REJECT-MISSING-COPY: tie
; REJECT-IMPLICIT-GROUP: fixed
; NUMBER: writable LReg
; CALL: call has no verified SFPU preservation ABI
; OFFSET: bound Dst offset must be defined and non-poison
; EXECUTOR: bound SFPU execution requires a TRISC executor

;--- unbound.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
define void @unbound() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  ret void
}
;--- tie.ll
declare void @llvm.riscv.tt.bound.sfpmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @tie() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 1, i32 0, i32 1, i32 0)
  ret void
}
;--- group.ll
declare void @llvm.riscv.tt.bound.sfplut(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @group() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfplut(i32 4, i32 4, i32 0, i32 1, i32 7, i32 3, i32 0)
  ret void
}
;--- number.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
define void @number() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 8, i32 9)
  ret void
}
;--- call.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @unknown()
define void @call() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @unknown()
  ret void
}
;--- offset.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @offset(i10 %address) "tensix-executor"="trisc1" {
  %index = zext i10 %address to i32
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %index, i32 0, i32 3)
  ret void
}
;--- executor.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
define void @executor() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  ret void
}
