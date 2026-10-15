; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/feature.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu %t/no-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu %t/undefined.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu %t/reserved.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESERVED
; FEATURE: SFU offline code generation requires +xboscsfu
; SCOPE: intrinsic requires trex-bosc-bound-rvv=1
; UNDEFINED: read of undefined physical vector register v4
; RESERVED: physical vector register must be in v1..v31; v0 is reserved

;--- feature.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.sfu.exp2(i32 immarg, i32 immarg)
define void @missing_feature(ptr %source) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %source)
  call void @llvm.riscv.bosc.sfu.exp2(i32 5, i32 4)
  ret void
}

;--- no-scope.ll
declare void @llvm.riscv.bosc.sfu.exp2(i32 immarg, i32 immarg)
define void @no_scope() {
  call void @llvm.riscv.bosc.sfu.exp2(i32 5, i32 4)
  ret void
}

;--- undefined.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.sfu.rcp(i32 immarg, i32 immarg)
define void @undefined() "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.sfu.rcp(i32 5, i32 4)
  ret void
}

;--- reserved.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.sfu.exp2(i32 immarg, i32 immarg)
define void @reserved() "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.sfu.exp2(i32 0, i32 4)
  ret void
}
