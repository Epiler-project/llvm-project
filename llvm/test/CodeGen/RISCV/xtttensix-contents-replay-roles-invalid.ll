; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/record-only.ll > %t/record-only-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/record-only-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RECORD-ONLY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/record-only-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RECORD-ONLY
; RUN: cat %t/common.ll %t/end-order.ll > %t/end-order-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/end-order-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=END-ORDER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/end-order-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=END-ORDER
; RUN: cat %t/common.ll %t/stale-body.ll > %t/stale-body-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/stale-body-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STALE-BODY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/stale-body-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STALE-BODY
; RUN: cat %t/common.ll %t/tuple-count.ll > %t/tuple-count-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/tuple-count-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TUPLE-COUNT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/tuple-count-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TUPLE-COUNT
; RUN: cat %t/common.ll %t/flat-type.ll > %t/flat-type-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/flat-type-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLAT-TYPE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/flat-type-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLAT-TYPE
; RUN: cat %t/common.ll %t/wrong-field.ll > %t/wrong-field-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wrong-field-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRONG-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/wrong-field-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRONG-FIELD
; RUN: cat %t/common.ll %t/execute-count.ll > %t/execute-count-input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/execute-count-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTE-COUNT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/execute-count-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTE-COUNT
; RECORD-ONLY: contents record-only body requires a non-executing recipe
; END-ORDER: contents replay end requires its actual final contents
; STALE-BODY: contents replay requires current mutable contents and declared read-only captures
; TUPLE-COUNT: contents replay execute requires an unpacked literal tuple matching mutable count
; FLAT-TYPE: contents replay roles require flat full contents values
; WRONG-FIELD: contents replay execute requires its current mutable contents
; EXECUTE-COUNT: contents replay execute mutable count differs from its record

;--- common.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 immarg, i32 immarg, ...)

;--- record-only.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  ret void
}

;--- end-order.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %checksum.first, <32 x i32> %sum.first)
  ret void
}

;--- stale-body.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %stale = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %sum, <32 x i32> %data)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  ret void
}

;--- tuple-count.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 62, i32 2, <32 x i32> %sum.first, <32 x i32> %checksum.first, <32 x i32> %data)
  ret void
}

;--- flat-type.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 62, i32 2, <32 x i32> %sum.first, <32 x i32> %checksum.first, i32 7)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  ret void
}

;--- wrong-field.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 62, i32 2, <32 x i32> %sum.first, <32 x i32> %checksum.first, <32 x i32> %data)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  %again = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 62, i32 2, <32 x i32> %checksum.next, <32 x i32> %sum.next, <32 x i32> %data)
  ret void
}

;--- execute-count.ll
define void @invalid() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 62, i32 1, <32 x i32> %sum.first, <32 x i32> %checksum.first, <32 x i32> %data)
  ret void
}
