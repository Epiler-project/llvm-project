; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/raw-record-only.ll -o %t/raw-bh.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/raw-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/raw-record-only.ll -o %t/raw-bh-o2.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/raw-bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/raw-record-execute.ll -o %t/execute-bh.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/execute-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/raw-record-execute.ll -o %t/execute-bh-o0.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/execute-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/structured.ll -o %t/structured-bh.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/structured-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/structured.ll -o %t/structured-wh.o 2>&1 | FileCheck %s --check-prefix=RECORD
; RUN: not test -e %t/structured-wh.o

; A no-old full definition has no declared mutable replay role. Until that
; contract is represented, reject recording before native/object publication;
; do not silently use the ordinary destructive LOAD's old/result projection.
; Raw LLVM REPLAY is already unavailable on WH; its structured contents owner
; is checked on both architectures instead of broadening that native surface.
; RECORD: contents full LOAD is not supported in replay recording

;--- raw-record-only.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
define void @recording_does_not_create_a_numerical_value() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  ret void
}

;--- raw-record-execute.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @record_and_execute_needs_a_defined_role_contract() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 0)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- structured.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
define void @full_definition_cannot_impersonate_a_mutable_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 17, i32 1, i32 0, i32 0, i32 1, <32 x i32> %initial)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 4, i32 0, i32 3)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 17, <32 x i32> %value)
  ret void
}
