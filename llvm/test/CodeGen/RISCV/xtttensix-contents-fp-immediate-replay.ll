; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s
; CHECK-LABEL: name: arithmetic_record
; CHECK: PseudoTTContentsSFPUReplay 1, 1,
; CHECK: TTSFPDIVP2
; CHECK: TTSFPADDI
; CHECK: TTSFPMULI
; CHECK: PseudoTTContentsSFPUReplay 0, 0,

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @arithmetic_record() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %exponent = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %old, i32 1, i32 1)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %exponent, i32 16256, i32 0)
  %mul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %add, i32 16384, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %mul)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %mul)
  %result = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

; DIVP2 also accepts the explicitly declared read-only contents capture.
; The immediate operations must advance that same mutable result role.
define void @arithmetic_record_capture() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: arithmetic_record_capture
; CHECK: PseudoTTContentsSFPUReplay 1, 1,
; CHECK: TTSFPDIVP2
; CHECK: TTSFPADDI
; CHECK: TTSFPMULI
; CHECK: PseudoTTContentsSFPUReplay 0, 0,
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 72, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %exponent = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %capture, i32 1, i32 1)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %exponent, i32 16256, i32 0)
  %mul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %add, i32 16384, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 72, <32 x i32> %mul)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 72, i32 1, <32 x i32> %mul, <32 x i32> %capture)
  %result = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}
