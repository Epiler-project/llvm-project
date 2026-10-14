; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/undefined.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/no-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/unbound.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNBOUND
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/virtual.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VIRTUAL
; UNDEFINED: read of undefined physical vector register v4 covering the active setvl lanes
; CONFIG: vector operation requires an explicit setvl on every incoming path
; UNBOUND: intrinsic requires trex-bosc-bound-rvv=1
; VIRTUAL: physical vector register must be in v1..v31; v0 is reserved
;--- undefined.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.fdiv(i32 immarg, i32 immarg, i32 immarg)
define void @undefined() "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.fdiv(i32 5, i32 4, i32 4)
  ret void
}
;--- no-config.ll
declare void @llvm.riscv.bosc.fsub(i32 immarg, i32 immarg, i32 immarg)
define void @no_config() "trex-bosc-bound-rvv"="1" {
  call void @llvm.riscv.bosc.fsub(i32 5, i32 4, i32 4)
  ret void
}
;--- unbound.ll
declare void @llvm.riscv.bosc.fmul(i32 immarg, i32 immarg, i32 immarg)
define void @unbound() {
  call void @llvm.riscv.bosc.fmul(i32 5, i32 4, i32 4)
  ret void
}
;--- virtual.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.fadd(i32 immarg, i32 immarg, i32 immarg)
define void @virtual() "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.fadd(i32 5, i32 0, i32 4)
  ret void
}
