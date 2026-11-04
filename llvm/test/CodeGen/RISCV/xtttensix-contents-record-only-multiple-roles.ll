; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.bh.O0.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.bh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.bh.O0.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only-roles.py --assembly %t.bh.O0.s --mir %t.bh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.bh.O2.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.bh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.bh.O2.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only-roles.py --assembly %t.bh.O2.s --mir %t.bh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.wh.O0.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.wh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o %t.wh.O0.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only-roles.py --assembly %t.wh.O0.s --mir %t.wh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.wh.O2.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.wh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o %t.wh.O2.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only-roles.py --assembly %t.wh.O2.s --mir %t.wh.O2.mir
;
; Flat role indices are independent of physical SFPR numbering. All five
; recipes retain mode0 nonexecuting recording, then execute's existing tuple
; defs and early-clobber old ties. A source mutable reads the earlier word's
; updated runtime contents. The dynamic loop has two ordinary PHIs and its
; zero-trip path; no source iteration is duplicated.
;
; VREG-LABEL: name:{{ *}}record_two_mutable
; VREG: [[SUM:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; VREG: [[CHECKSUM:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; VREG: [[DATA:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; VREG: PseudoTTContentsReplayTemplateBegin 162, 0, 0, 0, 2, [[SUM]], [[CHECKSUM]], [[DATA]],
; VREG-NEXT: PseudoTTContentsReplayTemplateRecordIAdd 162, 0, 2, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG-NEXT: PseudoTTContentsReplayTemplateRecordXor 162, 1, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG-NEXT: PseudoTTContentsReplayTemplateRecordNot 162, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG-NEXT: PseudoTTContentsReplayTemplateEnd 162, [[SUM]], [[CHECKSUM]],
; VREG: TTSFPSTORE [[SUM]], 0, 0, 4,
; VREG: TTSFPSTORE [[CHECKSUM]], 8, 0, 4,
; VREG: early-clobber [[NEXTSUM:%[0-9]+]]:sfpr, early-clobber [[NEXTCHECKSUM:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute2 162, 2, [[SUM]], [[CHECKSUM]], {{(killed )?}}[[DATA]],
; VREG: TTSFPSTORE {{(killed )?}}[[NEXTSUM]], 16, 0, 4,
; VREG: TTSFPSTORE {{(killed )?}}[[NEXTCHECKSUM]], 24, 0, 4,
; VREG-LABEL: name:{{ *}}record_two_mutable_loop
; VREG: PseudoTTContentsReplayTemplateBegin 163, 0, 0, 0, 2,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 163, 0, 2,
; VREG: PseudoTTContentsReplayTemplateRecordXor 163, 1, 0,
; VREG: :sfpr = PHI
; VREG: :sfpr = PHI
; VREG: early-clobber {{%[0-9]+}}:sfpr, early-clobber {{%[0-9]+}}:sfpr = PseudoTTContentsReplayTemplateExecute2 163, 2,
; VREG-LABEL: name:{{ *}}record_two_capture
; VREG: PseudoTTContentsReplayTemplateRecordAnd 164, 0, 2,
; VREG: PseudoTTContentsReplayTemplateRecordOr 164, 0, 1,
; VREG-LABEL: name:{{ *}}record_no_capture
; VREG: PseudoTTContentsReplayTemplateRecordNot 165, 1,
; VREG: PseudoTTContentsReplayTemplateRecordXor 165, 0, 1,
; PHYS-LABEL: name:{{ *}}record_two_mutable
; PHYS: PseudoTTContentsSFPUReplay 1, 0,
; PHYS: PseudoTTSFPURecordWord
; PHYS: PseudoTTReplayRecordEnd
; PHYS: TTSFPSTORE
; PHYS: TTSFPSTORE
; PHYS: PseudoTTContentsSFPUReplay 0, 0,
; PHYS: TTSFPSTORE
; PHYS: TTSFPSTORE
; PHYS-LABEL: name:{{ *}}record_two_mutable_loop
; PHYS: PseudoTTContentsSFPUReplay 1, 0,
; PHYS: PseudoTTReplayRecordEnd
; PHYS: PseudoTTContentsSFPUReplay 0, 0,
; PHYS: {{BNE|BEQ|BLT|BGE|BLTU|BGEU|PseudoBR}} {{.*}}%bb.

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

define void @record_two_mutable() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 162, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 162, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 162, i32 1, i32 0)
  call void @llvm.riscv.tt.contents.replay.template.record.not(i32 162, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 162, <32 x i32> %sum, <32 x i32> %checksum)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum, i32 8, i32 0, i32 4)
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 162, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum.next, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum.next, i32 24, i32 0, i32 4)
  ret void
}

define void @record_two_mutable_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %data = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 163, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum, <32 x i32> %data)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 163, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 163, i32 1, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 163, <32 x i32> %sum, <32 x i32> %checksum)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %remaining = phi i32 [ %count, %entry ], [ %decrement, %loop ]
  %sum.current = phi <32 x i32> [ %sum, %entry ], [ %sum.next, %loop ]
  %checksum.current = phi <32 x i32> [ %checksum, %entry ], [ %checksum.next, %loop ]
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 163, i32 2, <32 x i32> %sum.current, <32 x i32> %checksum.current, <32 x i32> %data)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  %decrement = sub i32 %remaining, 1
  %more = icmp ne i32 %decrement, 0
  br i1 %more, label %loop, label %exit
exit:
  %sum.result = phi <32 x i32> [ %sum, %entry ], [ %sum.next, %loop ]
  %checksum.result = phi <32 x i32> [ %checksum, %entry ], [ %checksum.next, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum.result, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum.result, i32 8, i32 0, i32 4)
  ret void
}

define void @record_two_capture() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 164, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture0, <32 x i32> %capture1)
  call void @llvm.riscv.tt.contents.replay.template.record.and(i32 164, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.replay.template.record.or(i32 164, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 164, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 164, i32 1, <32 x i32> %old, <32 x i32> %capture0, <32 x i32> %capture1)
  %next = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 8, i32 0, i32 4)
  ret void
}

define void @record_no_capture() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %checksum = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 165, i32 0, i32 0, i32 0, i32 2, <32 x i32> %sum, <32 x i32> %checksum)
  call void @llvm.riscv.tt.contents.replay.template.record.not(i32 165, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 165, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 165, <32 x i32> %sum, <32 x i32> %checksum)
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32s(i32 165, i32 2, <32 x i32> %sum, <32 x i32> %checksum)
  %sum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %checksum.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum.next, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %checksum.next, i32 8, i32 0, i32 4)
  ret void
}
