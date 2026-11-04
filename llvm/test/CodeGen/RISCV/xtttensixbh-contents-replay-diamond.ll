; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.O0.mir
; RUN: %python %S/Inputs/check-tensix-contents-replay-phi.py --mir %t.O0.mir --kind diamond
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.O2.mir
; RUN: %python %S/Inputs/check-tensix-contents-replay-phi.py --mir %t.O2.mir --kind diamond
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.O2.o
;
; Record once. Preserve the real CFG and first-result snapshot. Final replay
; must use and define the same mutable physical role and use the recorded
; readonly role, without emitting a repair copy/spill or cloning iterations.
; This is LLVM receiver coverage; it makes no hardware/source cutover claim.
;
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @contents_replay_diamond(i1 %run_once) "tensix-executor"="trisc1" {
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
  br i1 %run_once, label %one, label %merge

one:
  %once.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %once = extractvalue { <32 x i32> } %once.replay, 0
  br label %merge

merge:
  %current = phi <32 x i32> [ %first, %entry ], [ %once, %one ]
  %updated.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %current, <32 x i32> %capture)
  %updated = extractvalue { <32 x i32> } %updated.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %updated, i32 8, i32 0, i32 4)
  ret void
}
