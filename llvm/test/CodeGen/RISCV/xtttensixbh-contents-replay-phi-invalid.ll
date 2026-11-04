; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/foreign-seed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FOREIGNSEED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/foreign-seed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FOREIGNSEED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/changed-capture.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/changed-capture.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHANGEDCAPTURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/retained-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/retained-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETAINEDOLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/pre-record-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRERECORDOLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/pre-record-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRERECORDOLD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/foreign-phi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FOREIGNPHI
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/foreign-phi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FOREIGNPHI
;
; Every case keeps the machine verifier enabled. No crash or generic failure
; satisfies the exact owning diagnostic. Retained-old reaches real RA.
;
; FOREIGNSEED: contents replay execute requires its current mutable contents
; CHANGEDCAPTURE: contents replay execute requires its recorded read-only capture
; RETAINEDOLD: Tensix SFPU cannot synthesize a register copy
; PRERECORDOLD: contents replay execute requires its current mutable contents
; FOREIGNPHI: contents replay execute requires its current mutable contents

;--- foreign-seed.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @foreign_seed(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ; Observe actual first execution before any later destructive execution.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %foreign = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 14)
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %foreign, <32 x i32> %capture)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}

;--- changed-capture.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @changed_capture(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ; Observe actual first execution before any later destructive execution.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %changed = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %capture, <32 x i32> %capture, i32 4)
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %changed)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}

;--- retained-old.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @retained_old(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ; Observe actual first execution before any later destructive execution.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  ; Keeping the old contents live requires an unsupported numerical copy.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}

;--- pre-record-old.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @pre_record_old(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ; Observe actual first execution before any later destructive execution.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}

;--- foreign-phi.ll
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @foreign_phi(i1 %run_once) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ; Observe actual first execution before any later destructive execution.
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %foreign = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 14)
  br i1 %run_once, label %one, label %merge

one:
  %once.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %once = extractvalue { <32 x i32> } %once.replay, 0
  br label %merge

merge:
  %current = phi <32 x i32> [ %first, %entry ], [ %foreign, %one ]
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %current, <32 x i32> %capture)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}
