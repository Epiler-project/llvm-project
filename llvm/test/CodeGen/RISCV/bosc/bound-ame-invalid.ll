; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/no-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/no-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/undefined.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/role.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROLE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/stride.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; SCOPE: intrinsic requires trex-bosc-bound-ame=1
; CONFIG: operation requires M/N/K configuration
; UNDEFINED: read of undefined A tile register
; ROLE: read of undefined B tile register
; STRIDE: byte stride must be a constant covering the row

;--- no-scope.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
define i64 @no_scope(i64 %n) {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %n)
  ret i64 %m
}

;--- no-config.ll
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
define void @no_config(ptr %a) "trex-bosc-bound-ame"="1" {
  call void @llvm.riscv.bosc.ame.load.a(i32 1, ptr %a, i64 4)
  ret void
}

;--- undefined.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.mma(i32 immarg, i32 immarg, i32 immarg)
define void @undefined() "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.mma(i32 0, i32 1, i32 2)
  ret void
}

;--- role.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma(i32 immarg, i32 immarg, i32 immarg)
define void @role(ptr %a, ptr %b) "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.load.a(i32 1, ptr %a, i64 4)
  call void @llvm.riscv.bosc.ame.mma(i32 0, i32 1, i32 2)
  ret void
}

;--- stride.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
define void @stride(ptr %a) "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.load.a(i32 1, ptr %a, i64 1)
  ret void
}
