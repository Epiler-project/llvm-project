; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/no-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/no-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/undefined.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/reserved.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESERVED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/vector.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/asm.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl128b %t/feature.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/reconfigure.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDEFINED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; UNKNOWN: unknown BOSC intrinsic
; SCOPE: intrinsic requires trex-bosc-bound-rvv=1
; CONFIG: vector operation requires an explicit setvl on every incoming path
; UNDEFINED: read of undefined physical vector register v4
; RESERVED: physical vector register must be in v1..v31; v0 is reserved
; VECTOR: LLVM vector values are forbidden in bound scope
; CALL: calls, inline assembly and unbound intrinsics are forbidden in bound scope
; FEATURE: requires bound scope and RV64 +v +zvl512b

;--- no-scope.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
define i64 @no_scope(i64 %n) {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %n)
  ret i64 %vl
}
;--- no-config.ll
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
define void @no_config(ptr %p) "trex-bosc-bound-rvv"="1" {
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  ret void
}
;--- undefined.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @undefined(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- reserved.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
define void @reserved(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 0, ptr %p)
  ret void
}
;--- vector.ll
define <4 x i32> @vector(<4 x i32> %a) "trex-bosc-bound-rvv"="1" {
  ret <4 x i32> %a
}
;--- call.ll
declare void @external()
define void @call() "trex-bosc-bound-rvv"="1" {
  call void @external()
  ret void
}
;--- asm.ll
define void @asm() "trex-bosc-bound-rvv"="1" {
  call void asm sideeffect "", ""()
  ret void
}
;--- feature.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
define i64 @feature(i64 %n) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %n)
  ret i64 %vl
}
;--- join.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @join(ptr %p, i1 %cond) "trex-bosc-bound-rvv"="1" {
entry:
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  br i1 %cond, label %load, label %join
load:
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br label %join
join:
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- reconfigure.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @reconfigure(ptr %p, i64 %n) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %newvl = call i64 @llvm.riscv.bosc.setvl(i64 %n)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}

;--- unknown.ll
declare void @llvm.riscv.bosc.invalid()
define void @unknown() {
  call void @llvm.riscv.bosc.invalid()
  ret void
}
