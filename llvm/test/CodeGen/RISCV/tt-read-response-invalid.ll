; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+m -O0 < %t/rv64.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=RV64 --implicit-check-not='Stack dump:'
; RUN: not llc -mtriple=riscv64 -mattr=+m -O2 < %t/rv64.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=RV64 --implicit-check-not='Stack dump:'
; RUN: not llc -mtriple=riscv64 -mattr=+m -enable-new-pm \
; RUN:   < %t/rv64.ll 2>&1 | FileCheck %s --check-prefix=RV64 \
; RUN:   --implicit-check-not='Stack dump:'
; RUN: not llc -mtriple=riscv32 -mattr=+m < %t/bundle.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=DIRECT
; RUN: not llc -mtriple=riscv32 -mattr=+m \
; RUN:   < %t/calling-convention.ll 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: not llc -mtriple=riscv32 -mattr=+m -O0 < %t/memory-none.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m -O2 < %t/memory-none.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m -enable-new-pm \
; RUN:   < %t/memory-none.ll 2>&1 | FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m < %t/memory-read.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m < %t/memory-write.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m < %t/memory-argmem.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+m < %t/memory-inaccessiblemem.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MEMORY
; RUN: not opt -passes=verify -disable-output < %t/return.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=RETURN
; RUN: not opt -passes=verify -disable-output < %t/argument.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ARGUMENT
; RUN: not opt -passes=verify -disable-output < %t/arity.ll 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ARITY
; RV64: scalar device read response requires RV32
; DIRECT: scalar device read response requires a direct C call without operand bundles
; MEMORY: scalar device read response requires full memory effects
; RETURN: intrinsic return type expected void, but got i32
; ARGUMENT: intrinsic argument 0 type expected i32, but got i64
; ARITY: intrinsic has incorrect number of args. Expected 1, but got 2

;--- rv64.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(ptr %address) {
  %value = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.read.response(i32 %value)
  ret void
}

;--- bundle.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) [ "deopt"(i32 %value) ]
  ret void
}

;--- calling-convention.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call fastcc void @llvm.riscv.tt.read.response(i32 %value)
  ret void
}

;--- return.ll
declare i32 @llvm.riscv.tt.read.response(i32)
define i32 @bad(i32 %value) {
  %result = call i32 @llvm.riscv.tt.read.response(i32 %value)
  ret i32 %result
}

; Call-site effects are not normalized to the intrinsic's declaration effects.
; A scalar response handoff must preserve memory actions in every region.

;--- memory-none.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) memory(none)
  ret void
}

;--- memory-read.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) memory(read)
  ret void
}

;--- memory-write.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) memory(write)
  ret void
}

;--- memory-argmem.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) memory(argmem: readwrite)
  ret void
}

;--- memory-inaccessiblemem.ll
declare void @llvm.riscv.tt.read.response(i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value) memory(inaccessiblemem: readwrite)
  ret void
}

;--- argument.ll
declare void @llvm.riscv.tt.read.response(i64)
define void @bad(i64 %value) {
  call void @llvm.riscv.tt.read.response(i64 %value)
  ret void
}

;--- arity.ll
declare void @llvm.riscv.tt.read.response(i32, i32)
define void @bad(i32 %value) {
  call void @llvm.riscv.tt.read.response(i32 %value, i32 %value)
  ret void
}
