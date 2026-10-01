; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: FileCheck %s < %t.o0.s
; RUN: FileCheck %s < %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o2.mir
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o0.s %t.o0.mir record_only record_and_execute
; RUN: %python %S/Inputs/tensix-explicit-replay-observer.py %t.o2.s %t.o2.mir record_only record_and_execute
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=EARLY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=EARLY
;
; The structured handoff has no length operand. IDs 41 and 97 are compiler
; identities, not slots; both independently constrained functions use slot 5.
; A wrongly executing record-only body changes the first observation. Capturing
; L1 contents instead of its physical identity changes the second observation.
; The existing independent emitted-word observer also rejects added SFPU work,
; virtual SFPRs, and incorrect numerical effects. This is software codegen
; evidence; it does not execute a device or establish general Dst semantics.
;
; begin arguments: identity, mode (0 record-only / 1 record-and-execute),
; placement (0 automatic / 1 fixed), fixed start. Only direct local issue is
; exercised by this first handoff fixture.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @record_only() "tensix-executor"="trisc1" {
; EARLY-LABEL: name: record_only
; EARLY: PseudoTTReplayTemplateBegin 41, 0, 1, 5,
; EARLY-NEXT: PseudoTTReplayTemplateWord 41,
; EARLY-NOT: $tt_l0
; EARLY-NEXT: PseudoTTReplayTemplateEnd 41,
; EARLY: PseudoTTReplayTemplateExecute 41,
; EARLY-SAME: implicit $tt_l1
; EARLY-SAME: implicit-def $tt_l0
; CHECK-LABEL: record_only:
; CHECK: .word 0xf0002809
; CHECK: .word 0xf0000849
; Slot5/derived length1, load=1 and execute-while-loading=0.
; CHECK: .word 0x10050044
; CHECK-NEXT: .word 0xf0000409
; CHECK: .word 0xc8100001
; CHECK: .word 0xf0002449
; CHECK: .word 0x10050040
; CHECK: .word 0xc8100021
; CHECK: ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.replay.template.begin(i32 41, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 41)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay.template.execute(i32 41)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  ret void
}

define void @record_and_execute() "tensix-executor"="trisc1" {
; EARLY-LABEL: name: record_and_execute
; EARLY: PseudoTTReplayTemplateBegin 97, 1, 1, 5,
; EARLY-NEXT: $tt_l0 = TTSFPMOVAll $tt_l1,
; EARLY-NEXT: PseudoTTReplayTemplateEnd 97,
; EARLY: PseudoTTReplayTemplateExecute 97,
; EARLY-SAME: implicit $tt_l1
; EARLY-SAME: implicit-def $tt_l0
; CHECK-LABEL: record_and_execute:
; CHECK: .word 0xf0002809
; CHECK: .word 0xf0000849
; The mode bit differs; no additional execute implements the first execution.
; CHECK: .word 0x1005004c
; CHECK-NEXT: .word 0xf0000409
; CHECK: .word 0xc8100001
; CHECK: .word 0xf0002449
; CHECK: .word 0x10050040
; CHECK: .word 0xc8100021
; CHECK: ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.replay.template.begin(i32 97, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 97)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay.template.execute(i32 97)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  ret void
}
