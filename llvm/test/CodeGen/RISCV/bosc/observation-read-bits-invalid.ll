; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/negative-width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/wide-width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/zero-width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/dynamic-width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC-WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/no-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/zero.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REG
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REG
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/clobber.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CLOBBER
; RUN: not llc -mtriple=riscv32 -mattr=+v,+zvl512b %t/zero-vl.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV64
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/raw-avl.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ACTUAL
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/unknown-nonzero.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=JOIN
; SCOPE: bit observation requires trex-bosc-bound-rvv=1
; REG: physical vector register must be in v1..v31
; CLOBBER: read of undefined physical vector register v4 covering the active setvl lanes
; RV64: requires bound scope and RV64 +v +zvl512b
; ACTUAL: bit observation must restore the active setvl result
; UNKNOWN: bit observation without a source configuration must restore literal zero
; JOIN: bit observation requires one reaching vector configuration
;
;--- no-scope.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @no_scope(ptr %p) {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 8)
  ret i64 %v
}
;--- zero.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @zero(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 0, ptr %p, i64 0, i32 8)
  ret i64 %v
}
;--- range.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @range(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 32, ptr %p, i64 0, i32 8)
  ret i64 %v
}
;--- clobber.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @clobber(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 %vl, i32 8)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- zero-vl.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @zero_vl(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 8)
  ret i64 %v
}
;--- raw-avl.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
declare i64 @llvm.riscv.bosc.setvl(i64)
define i64 @raw_avl(ptr %p, i64 %avl) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 %avl, i32 8)
  ret i64 %v
}
;--- unknown-nonzero.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @unknown_nonzero(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 7, i32 8)
  ret i64 %v
}
;--- join.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
declare i64 @llvm.riscv.bosc.setvl(i64)
define i64 @join(ptr %p, i1 %condition) "trex-bosc-bound-rvv"="1" {
entry:
  br i1 %condition, label %left, label %right
left:
  %lv = call i64 @llvm.riscv.bosc.setvl(i64 7)
  br label %merge
right:
  %rv = call i64 @llvm.riscv.bosc.setvl(i64 16)
  br label %merge
merge:
  %vl = phi i64 [ %lv, %left ], [ %rv, %right ]
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 %vl, i32 8)
  ret i64 %v
}

; WIDTH: bit observation width must be 8, 32, or 64
; DYNAMIC-WIDTH: immarg operand has non-immediate parameter
;--- width.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @width(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 16)
  ret i64 %v
}
;--- dynamic-width.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @dynamic_width(ptr %p, i32 %width) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 %width)
  ret i64 %v
}
;--- zero-width.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @bad_width(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 0)
  ret i64 %v
}
;--- wide-width.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @bad_width(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 128)
  ret i64 %v
}
;--- negative-width.ll
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
define i64 @bad_width(ptr %p) "trex-bosc-bound-rvv"="1" {
  %v = call i64 @llvm.riscv.bosc.read.bits(i32 4, ptr %p, i64 0, i32 -1)
  ret i64 %v
}
