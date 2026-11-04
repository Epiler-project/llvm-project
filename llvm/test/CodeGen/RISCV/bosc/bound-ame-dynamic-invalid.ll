; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/wrapped-stride.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; UNKNOWN: dimension request must be provably positive and nonwrapping
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/zero.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ZERO
; ZERO: dimension request must be provably positive and nonwrapping
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/stride.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/wrong-dimension.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 %t/wrong-dimension.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; STRIDE: byte stride must be provably bounded covering the row
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/alignment.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALIGNMENT
; ALIGNMENT: accumulator stride must be 4-byte aligned
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=JOIN
; JOIN: matching dimensions on every incoming path

;--- unknown.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @unknown(i64 %raw) "trex-bosc-bound-ame"="1" {
    %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %raw)
  ret void
}

;--- zero.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @zero(i64 %raw) "trex-bosc-bound-ame"="1" {
  %request = and i64 %raw, 63
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %request)
  ret void
}

;--- stride.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @stride(ptr %p, i64 %raw) "trex-bosc-bound-ame"="1" {
  %pad = and i64 %raw, 252
  %stride = add i64 %pad, 252
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 64)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 64)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %p, i64 %stride)
  ret void
}

;--- alignment.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @alignment(ptr %p, i64 %raw) "trex-bosc-bound-ame"="1" {
  %pad = and i64 %raw, 253
  %stride = add i64 %pad, 256
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 64)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 64)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %p, i64 %stride)
  ret void
}

;--- join.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @join(ptr %p, i64 %x, i64 %y, i1 %choose) "trex-bosc-bound-ame"="1" {
  %a0 = and i64 %x, 63
  %a = add i64 %a0, 1
  %b0 = and i64 %y, 63
  %b = add i64 %b0, 1
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 64)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  br i1 %choose, label %left, label %right
left:
  %m0 = call i64 @llvm.riscv.bosc.ame.setm(i64 %a)
  br label %merge
right:
  %m1 = call i64 @llvm.riscv.bosc.ame.setm(i64 %b)
  br label %merge
merge:
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %p, i64 256)
  ret void
}

;--- wrapped-stride.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
define void @wrapped_stride(ptr %p, i64 %raw) "trex-bosc-bound-ame"="1" {
  %low = and i64 %raw, 63
  %request = add i64 %low, 4611686018427387904
  %stride = mul i64 %request, 4
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %request)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %request)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %request)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %p, i64 %stride)
  ret void
}

;--- wrong-dimension.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
; Matching bounds do not imply matching values: M can be smaller than N.
define void @wrong_actual_dimension_stride(ptr %p, i64 %rows, i64 %columns) "trex-bosc-bound-ame"="1" {
  %mlow = and i64 %rows, 127
  %nlow = and i64 %columns, 127
  %requested_m = add i64 %mlow, 1
  %requested_n = add i64 %nlow, 1
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %requested_m)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %requested_n)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 1)
  %stride = mul i64 %m, 4
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %p, i64 %stride)
  ret void
}
