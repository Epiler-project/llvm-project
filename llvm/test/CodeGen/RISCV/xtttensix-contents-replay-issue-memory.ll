; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SOURCE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=SOURCE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=SOURCE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=SOURCE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %S/xtttensix-contents-mop.ll -o - | FileCheck %s --check-prefix=MOP
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %S/xtttensix-contents-mop.ll -o - | FileCheck %s --check-prefix=MOP
;
; Formal lowering attaches an unknown-pointer, volatile, load/store, aligned
; four-byte issue MMO to begin/execute. The receiver must retain it at both
; pre-RA and finalized control boundaries. This uses real IR lowering rather
; than accepting only synthetic MIR without a memory operand.
; SOURCE: PseudoTTContentsReplayTemplateBegin 41, 1, 0, 0, 1,
; SOURCE-SAME: :: (volatile load store (s32))
; SOURCE: early-clobber {{%[0-9]+}}:sfpr = PseudoTTContentsReplayTemplateExecute1 41, 1,
; SOURCE-SAME: :: (volatile load store (s32))
; FINAL: PseudoTTContentsSFPUReplay 1, 1,
; FINAL-SAME: :: (volatile load store (s32))
; FINAL: PseudoTTContentsSFPUReplay 0, 0,
; FINAL-SAME: :: (volatile load store (s32))
; MOP: PseudoTTContentsReplayTemplateBegin 71, 0, 0, 0, 1,
; MOP-SAME: :: (volatile load store (s32))
; MOP: PseudoTTContentsReplayTemplateMopExecute1 71, 1,
; MOP-SAME: :: (volatile load store (s32))

target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @issue_memory() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %next = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 0, i32 0, i32 4)
  ret void
}
