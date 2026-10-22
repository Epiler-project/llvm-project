; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/gt-no-effect.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GT-MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/gt-stack.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GT-MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/le-write.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LE-MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/le-stack-or.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LE-MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/gt-lhs.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REGISTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/le-backdoor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REGISTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/gt-rhs.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REGISTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/le-dynamic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: not llc -mtriple=riscv32 %t/feature.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/attribute.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/result.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESULT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/field-type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIELD-TYPE
;
; The receiver is condition-only: neither SET_VD nor stack mutation is a
; hidden result. VD is an LReg read, not a configurable backdoor register.
; Native fields remain immediate i32 operands and no numerical value ABI is
; admitted, including when the returned value would otherwise be unused.
; GT-MODE: SFPGT mode must be one of {1}
; LE-MODE: SFPLE mode must be one of {1}
; REGISTER: bound SFPU invalid read register operand class
; CONSTANT: immarg operand has non-immediate parameter
; FEATURE: Tensix intrinsic requires +xtttensixbh
; ATTRIBUTE: Tensix intrinsic requires a tensix-executor function attribute
; EXECUTOR: bound SFPU execution requires a TRISC executor
; RESULT: intrinsic return type expected void, but got i32
; FIELD-TYPE: intrinsic argument 0 type expected i32, but got i64

;--- gt-no-effect.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @gt_no_effect() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 0)
  ret void
}

;--- gt-stack.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @gt_stack() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 3)
  ret void
}

;--- le-write.ll
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @le_write() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 9)
  ret void
}

;--- le-stack-or.ll
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @le_stack_or() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 7)
  ret void
}

;--- gt-lhs.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @gt_lhs() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 8, i32 1, i32 1)
  ret void
}

;--- le-backdoor.ll
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @le_backdoor() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfple(i32 12, i32 1, i32 1)
  ret void
}

;--- gt-rhs.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @gt_rhs() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 16, i32 1)
  ret void
}

;--- le-dynamic.ll
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @le_dynamic(i32 %mode) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 %mode)
  ret void
}

;--- feature.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @feature() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 1)
  ret void
}

;--- attribute.ll
declare void @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @attribute() {
  call void @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 1)
  ret void
}

;--- executor.ll
declare void @llvm.riscv.tt.bound.sfpgt(i32 immarg, i32 immarg, i32 immarg)
define void @executor() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.bound.sfpgt(i32 0, i32 1, i32 1)
  ret void
}

;--- result.ll
declare i32 @llvm.riscv.tt.bound.sfple(i32 immarg, i32 immarg, i32 immarg)
define void @result() "tensix-executor"="trisc1" {
  %result = call i32 @llvm.riscv.tt.bound.sfple(i32 0, i32 1, i32 1)
  ret void
}

;--- field-type.ll
declare void @llvm.riscv.tt.bound.sfpgt(i64 immarg, i32 immarg, i32 immarg)
define void @field_type() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpgt(i64 0, i32 1, i32 1)
  ret void
}
