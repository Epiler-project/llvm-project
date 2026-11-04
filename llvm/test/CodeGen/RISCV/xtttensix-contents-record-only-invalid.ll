; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/orphan.ll > %t/orphan.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/orphan.input.ll -o %t/orphan.bh.O0.o 2>&1 | FileCheck %s --check-prefix=ORPHAN
; RUN: not test -e %t/orphan.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/orphan.input.ll -o %t/orphan.bh.O2.o 2>&1 | FileCheck %s --check-prefix=ORPHAN
; RUN: not test -e %t/orphan.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/orphan.input.ll -o %t/orphan.wh.O0.o 2>&1 | FileCheck %s --check-prefix=ORPHAN
; RUN: not test -e %t/orphan.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/orphan.input.ll -o %t/orphan.wh.O2.o 2>&1 | FileCheck %s --check-prefix=ORPHAN
; RUN: not test -e %t/orphan.wh.O2.o
; RUN: cat %t/common.ll %t/mode-one.ll > %t/mode-one.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/mode-one.input.ll -o %t/mode-one.bh.O0.o 2>&1 | FileCheck %s --check-prefix=MODEONE
; RUN: not test -e %t/mode-one.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/mode-one.input.ll -o %t/mode-one.bh.O2.o 2>&1 | FileCheck %s --check-prefix=MODEONE
; RUN: not test -e %t/mode-one.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/mode-one.input.ll -o %t/mode-one.wh.O0.o 2>&1 | FileCheck %s --check-prefix=MODEONE
; RUN: not test -e %t/mode-one.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/mode-one.input.ll -o %t/mode-one.wh.O2.o 2>&1 | FileCheck %s --check-prefix=MODEONE
; RUN: not test -e %t/mode-one.wh.O2.o
; RUN: cat %t/common.ll %t/wrong-identity.ll > %t/wrong-identity.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wrong-identity.input.ll -o %t/wrong-identity.bh.O0.o 2>&1 | FileCheck %s --check-prefix=WRONGIDENTITY
; RUN: not test -e %t/wrong-identity.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wrong-identity.input.ll -o %t/wrong-identity.bh.O2.o 2>&1 | FileCheck %s --check-prefix=WRONGIDENTITY
; RUN: not test -e %t/wrong-identity.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wrong-identity.input.ll -o %t/wrong-identity.wh.O0.o 2>&1 | FileCheck %s --check-prefix=WRONGIDENTITY
; RUN: not test -e %t/wrong-identity.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wrong-identity.input.ll -o %t/wrong-identity.wh.O2.o 2>&1 | FileCheck %s --check-prefix=WRONGIDENTITY
; RUN: not test -e %t/wrong-identity.wh.O2.o
; RUN: cat %t/common.ll %t/wrong-end.ll > %t/wrong-end.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wrong-end.input.ll -o %t/wrong-end.bh.O0.o 2>&1 | FileCheck %s --check-prefix=WRONGEND
; RUN: not test -e %t/wrong-end.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wrong-end.input.ll -o %t/wrong-end.bh.O2.o 2>&1 | FileCheck %s --check-prefix=WRONGEND
; RUN: not test -e %t/wrong-end.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wrong-end.input.ll -o %t/wrong-end.wh.O0.o 2>&1 | FileCheck %s --check-prefix=WRONGEND
; RUN: not test -e %t/wrong-end.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wrong-end.input.ll -o %t/wrong-end.wh.O2.o 2>&1 | FileCheck %s --check-prefix=WRONGEND
; RUN: not test -e %t/wrong-end.wh.O2.o
; RUN: cat %t/common.ll %t/changed-capture.ll > %t/changed-capture.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/changed-capture.input.ll -o %t/changed-capture.bh.O0.o 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not test -e %t/changed-capture.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/changed-capture.input.ll -o %t/changed-capture.bh.O2.o 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not test -e %t/changed-capture.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/changed-capture.input.ll -o %t/changed-capture.wh.O0.o 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not test -e %t/changed-capture.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/changed-capture.input.ll -o %t/changed-capture.wh.O2.o 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not test -e %t/changed-capture.wh.O2.o
; RUN: cat %t/common.ll %t/foreign-old.ll > %t/foreign-old.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/foreign-old.input.ll -o %t/foreign-old.bh.O0.o 2>&1 | FileCheck %s --check-prefix=FOREIGNOLD
; RUN: not test -e %t/foreign-old.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/foreign-old.input.ll -o %t/foreign-old.bh.O2.o 2>&1 | FileCheck %s --check-prefix=FOREIGNOLD
; RUN: not test -e %t/foreign-old.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/foreign-old.input.ll -o %t/foreign-old.wh.O0.o 2>&1 | FileCheck %s --check-prefix=FOREIGNOLD
; RUN: not test -e %t/foreign-old.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/foreign-old.input.ll -o %t/foreign-old.wh.O2.o 2>&1 | FileCheck %s --check-prefix=FOREIGNOLD
; RUN: not test -e %t/foreign-old.wh.O2.o
; RUN: cat %t/common.ll %t/nested.ll > %t/nested.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/nested.input.ll -o %t/nested.bh.O0.o 2>&1 | FileCheck %s --check-prefix=NESTED
; RUN: not test -e %t/nested.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/nested.input.ll -o %t/nested.bh.O2.o 2>&1 | FileCheck %s --check-prefix=NESTED
; RUN: not test -e %t/nested.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/nested.input.ll -o %t/nested.wh.O0.o 2>&1 | FileCheck %s --check-prefix=NESTED
; RUN: not test -e %t/nested.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/nested.input.ll -o %t/nested.wh.O2.o 2>&1 | FileCheck %s --check-prefix=NESTED
; RUN: not test -e %t/nested.wh.O2.o
; RUN: cat %t/common.ll %t/numerical-record.ll > %t/numerical-record.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/numerical-record.input.ll -o %t/numerical-record.bh.O0.o 2>&1 | FileCheck %s --check-prefix=NUMERICALRECORD
; RUN: not test -e %t/numerical-record.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/numerical-record.input.ll -o %t/numerical-record.bh.O2.o 2>&1 | FileCheck %s --check-prefix=NUMERICALRECORD
; RUN: not test -e %t/numerical-record.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/numerical-record.input.ll -o %t/numerical-record.wh.O0.o 2>&1 | FileCheck %s --check-prefix=NUMERICALRECORD
; RUN: not test -e %t/numerical-record.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/numerical-record.input.ll -o %t/numerical-record.wh.O2.o 2>&1 | FileCheck %s --check-prefix=NUMERICALRECORD
; RUN: not test -e %t/numerical-record.wh.O2.o
; RUN: cat %t/common.ll %t/retained-old.ll > %t/retained-old.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/retained-old.input.ll -o %t/retained-old.bh.O0.o 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not test -e %t/retained-old.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/retained-old.input.ll -o %t/retained-old.bh.O2.o 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not test -e %t/retained-old.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/retained-old.input.ll -o %t/retained-old.wh.O0.o 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not test -e %t/retained-old.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/retained-old.input.ll -o %t/retained-old.wh.O2.o 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not test -e %t/retained-old.wh.O2.o
; RUN: cat %t/common.ll %t/nondominating.ll > %t/nondominating.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/nondominating.input.ll -o %t/nondominating.bh.O0.o 2>&1 | FileCheck %s --check-prefix=NONDOMINATING
; RUN: not test -e %t/nondominating.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/nondominating.input.ll -o %t/nondominating.bh.O2.o 2>&1 | FileCheck %s --check-prefix=NONDOMINATING
; RUN: not test -e %t/nondominating.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/nondominating.input.ll -o %t/nondominating.wh.O0.o 2>&1 | FileCheck %s --check-prefix=NONDOMINATING
; RUN: not test -e %t/nondominating.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/nondominating.input.ll -o %t/nondominating.wh.O2.o 2>&1 | FileCheck %s --check-prefix=NONDOMINATING
; RUN: not test -e %t/nondominating.wh.O2.o
; RUN: cat %t/common.ll %t/fixed-overlap.ll > %t/fixed-overlap.input.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/fixed-overlap.input.ll -o %t/fixed-overlap.bh.O0.o 2>&1 | FileCheck %s --check-prefix=FIXEDOVERLAP
; RUN: not test -e %t/fixed-overlap.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/fixed-overlap.input.ll -o %t/fixed-overlap.bh.O2.o 2>&1 | FileCheck %s --check-prefix=FIXEDOVERLAP
; RUN: not test -e %t/fixed-overlap.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/fixed-overlap.input.ll -o %t/fixed-overlap.wh.O0.o 2>&1 | FileCheck %s --check-prefix=FIXEDOVERLAP
; RUN: not test -e %t/fixed-overlap.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/fixed-overlap.input.ll -o %t/fixed-overlap.wh.O2.o 2>&1 | FileCheck %s --check-prefix=FIXEDOVERLAP
; RUN: not test -e %t/fixed-overlap.wh.O2.o
;
; These failures retain the exact owner diagnostic and no object. An
; unrelated parser failure, silent fallback, or emitted partial object is
; insufficient. Retained-old reaches real RA; fixed-overlap reaches the
; native replay-slot lifetime/placement owner.
; ORPHAN: contents replay recipe requires its active record-only owner
; MODEONE: contents replay recipe requires its active record-only owner
; WRONGIDENTITY: contents replay recipe requires its active record-only owner
; WRONGEND: contents replay end requires its actual final contents
; CHANGEDCAPTURE: contents replay execute requires its recorded read-only capture
; FOREIGNOLD: contents replay execute requires its current mutable contents
; NESTED: contents replay requires one non-nested producer
; NUMERICALRECORD: contents record-only body requires a non-executing recipe
; RETAINEDOLD: Tensix SFPU cannot synthesize a register copy
; NONDOMINATING: contents replay execute requires a dominating record
; FIXEDOVERLAP: structured replay fixed placement conflicts or exceeds capacity

;--- common.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)

;--- orphan.ll
define void @orphan() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  ret void
}

;--- mode-one.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- wrong-identity.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 40, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- wrong-end.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- changed-capture.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %changed = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %capture, <32 x i32> %capture, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %changed)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- foreign-old.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %foreign = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %foreign, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- nested.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 42, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- numerical-record.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %fake = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %fake)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- retained-old.ll
define void @record_only() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 16, i32 0, i32 4)
  ret void
}

;--- nondominating.ll
define void @nondominating(i1 %guard) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  br i1 %guard, label %record, label %merge
record:
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 41, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %old)
  br label %merge
merge:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

;--- fixed-overlap.ll
define void @fixed_overlap() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 51, i32 0, i32 1, i32 0, i32 1, <32 x i32> %a, <32 x i32> %b)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 51, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 51, <32 x i32> %a)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 52, i32 0, i32 1, i32 0, i32 1, <32 x i32> %c, <32 x i32> %d)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 52, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 52, <32 x i32> %c)
  %ar.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 51, i32 1, <32 x i32> %a, <32 x i32> %b)
  %ar = extractvalue { <32 x i32> } %ar.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %ar, i32 0, i32 0, i32 4)
  %cr.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 52, i32 1, <32 x i32> %c, <32 x i32> %d)
  %cr = extractvalue { <32 x i32> } %cr.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %cr, i32 8, i32 0, i32 4)
  ret void
}
