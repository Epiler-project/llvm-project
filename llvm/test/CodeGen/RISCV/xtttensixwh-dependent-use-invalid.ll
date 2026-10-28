; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 < %t/no-feature.ll 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/no-executor.ll 2>&1 | FileCheck %s --check-prefix=NOEXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/bad-executor.ll 2>&1 | FileCheck %s --check-prefix=BADEXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/bundle.ll 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/calling-convention.ll 2>&1 | FileCheck %s --check-prefix=DIRECT
; FEATURE: Tensix intrinsic requires +xtttensix
; NOEXECUTOR: Tensix intrinsic requires a tensix-executor function attribute
; BADEXECUTOR: invalid tensix-executor value: host
; DIRECT: Tensix intrinsic requires a direct C call without operand bundles

; The scalar intrinsic keeps its feature, executor, and call-shape contract;
; admitting Wormhole must not turn it into an unrestricted generic intrinsic.

;--- no-feature.ll
declare void @llvm.riscv.tt.dependent.use(i32)
define void @bad(i32 %value) "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  ret void
}

;--- no-executor.ll
declare void @llvm.riscv.tt.dependent.use(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  ret void
}

;--- bad-executor.ll
declare void @llvm.riscv.tt.dependent.use(i32)
define void @bad(i32 %value) "tensix-executor"="host" {
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  ret void
}

;--- bundle.ll
declare void @llvm.riscv.tt.dependent.use(i32)
define void @bad(i32 %value) "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.dependent.use(i32 %value) [ "deopt"(i32 %value) ]
  ret void
}

;--- calling-convention.ll
declare void @llvm.riscv.tt.dependent.use(i32)
define void @bad(i32 %value) "tensix-executor"="ncrisc" {
  call fastcc void @llvm.riscv.tt.dependent.use(i32 %value)
  ret void
}
