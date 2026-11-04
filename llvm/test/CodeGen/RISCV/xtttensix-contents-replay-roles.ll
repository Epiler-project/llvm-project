; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
;
; Independent flat roles use independent SFPR vregs. A literal struct packs
; returns only; the dynamic loop retains two ordinary vector PHIs and its
; zero-trip path. No source iteration is enumerated or cloned.
; VREG-LABEL: name:{{ *}}two_captures
; VREG: PseudoTTContentsReplayTemplateBegin 61, 1, 0, 0, 1,
; VREG: early-clobber {{%[0-9]+}}:sfpr = PseudoTTContentsReplayTemplateExecute1 61, 1,
; VREG-LABEL: name:{{ *}}two_mutable_loop
; VREG: :sfpr = PHI
; VREG: :sfpr = PHI
; VREG: early-clobber {{%[0-9]+}}:sfpr, early-clobber {{%[0-9]+}}:sfpr = PseudoTTContentsReplayTemplateExecute2 62, 2,
; VREG-LABEL: name:{{ *}}updated_cross_role
; VREG: TTSFPIADD
; VREG: TTSFPXOR
; VREG-LABEL: name:{{ *}}no_capture
; VREG: PseudoTTContentsReplayTemplateBegin 63, 1, 0, 0, 1,
; PHYS-LABEL: name:{{ *}}two_captures
; PHYS: PseudoTTContentsSFPUReplay 1, 1,
; PHYS: TTSFPIADD
; PHYS: TTSFPXOR
; PHYS: PseudoTTContentsSFPUReplay 0, 0,
; PHYS-LABEL: name:{{ *}}two_mutable_loop
; PHYS: PseudoTTContentsSFPUReplay 1, 1,
; PHYS: TTSFPIADD
; PHYS: TTSFPXOR
; PHYS: PseudoTTContentsSFPUReplay 0, 0,
; PHYS: TTSFPSTORE
; PHYS: TTSFPSTORE

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

define void @two_captures() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 61, i32 1, i32 0, i32 0, i32 1, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %add = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %checksum, i32 4)
  %first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %add, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 61, <32 x i32> %first)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 61, i32 1, <32 x i32> %first, <32 x i32> %checksum, <32 x i32> %data)
  %next = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 0, i32 0, i32 4)
  ret void
}

define void @two_mutable_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %data)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %remaining = phi i32 [ %count, %entry ], [ %decrement, %loop ]
  %sum.current = phi <32 x i32> [ %sum.first, %entry ], [ %sum.next, %loop ]
  %checksum.current = phi <32 x i32> [ %checksum.first, %entry ], [ %checksum.next, %loop ]
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 62, i32 2, <32 x i32> %sum.current, <32 x i32> %checksum.current, <32 x i32> %data)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  %decrement = sub i32 %remaining, 1
  %done = icmp eq i32 %decrement, 0
  br i1 %done, label %exit, label %loop
exit:
  %sum.final = phi <32 x i32> [ %sum.first, %entry ], [ %sum.next, %loop ]
  %checksum.final = phi <32 x i32> [ %checksum.first, %entry ], [ %checksum.next, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum.final, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum.final, i32 8, i32 0, i32 4)
  ret void
}

define void @updated_cross_role() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 62, i32 1, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %sum, <32 x i32> %data, i32 4)
  %checksum.first = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %checksum, <32 x i32> %sum.first)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 62, <32 x i32> %sum.first, <32 x i32> %checksum.first)
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 62, i32 2, <32 x i32> %sum.first, <32 x i32> %checksum.first, <32 x i32> %data)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum.next, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum.next, i32 8, i32 0, i32 4)
  ret void
}

define void @no_capture() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 63, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %first = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 63, <32 x i32> %first)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 63, i32 1, <32 x i32> %first)
  %next = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 0, i32 0, i32 4)
  ret void
}
