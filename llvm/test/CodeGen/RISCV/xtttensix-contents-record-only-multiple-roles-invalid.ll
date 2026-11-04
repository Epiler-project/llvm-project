; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/destination-capture.ll > %t/destination-capture-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/destination-capture-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/destination-capture-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: cat %t/common.ll %t/destination-negative.ll > %t/destination-negative-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/destination-negative-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/destination-negative-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: cat %t/common.ll %t/source-undeclared.ll > %t/source-undeclared-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/source-undeclared-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SOURCE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/source-undeclared-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SOURCE
; RUN: cat %t/common.ll %t/source-negative.ll > %t/source-negative-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/source-negative-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SOURCE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/source-negative-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SOURCE
; RUN: cat %t/common.ll %t/unary-destination-capture.ll > %t/unary-destination-capture-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/unary-destination-capture-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/unary-destination-capture-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESTINATION
; RUN: cat %t/common.ll %t/destination-dynamic.ll > %t/destination-dynamic-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/destination-dynamic-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/destination-dynamic-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC
; RUN: cat %t/common.ll %t/source-dynamic.ll > %t/source-dynamic-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/source-dynamic-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/source-dynamic-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC
; DESTINATION: contents replay recipe destination requires a declared mutable role
; SOURCE: contents replay recipe source requires a declared contents role
; DYNAMIC: immarg operand has non-immediate parameter
;--- common.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.and(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.or(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.xor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.not(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 immarg, i32 immarg, ...)

declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
;--- destination-capture.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 162, i32 2, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- destination-negative.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 162, i32 -1, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- source-undeclared.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 162, i32 1, i32 3)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- source-negative.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 162, i32 1, i32 -1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- unary-destination-capture.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.not(i32 162, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- destination-dynamic.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 162, i32 %dynamic_role, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}

;--- source-dynamic.ll
define void @invalid(i32 %dynamic_role) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 162, i32 1, i32 %dynamic_role)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)

  ret void
}
