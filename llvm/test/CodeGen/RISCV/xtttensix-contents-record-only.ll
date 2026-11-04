; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/positive.ll > %t/input.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o %t.bh.O0.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.bh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/input.ll -o %t.bh.O0.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only.py --assembly %t.bh.O0.s --mir %t.bh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/input.ll -o %t.bh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o %t.bh.O2.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.bh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/input.ll -o %t.bh.O2.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only.py --assembly %t.bh.O2.s --mir %t.bh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/input.ll -o %t.bh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o %t.wh.O0.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.wh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/input.ll -o %t.wh.O0.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only.py --assembly %t.wh.O0.s --mir %t.wh.O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/input.ll -o %t.wh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o %t.wh.O2.mir
; RUN: FileCheck %s --check-prefix=PHYS < %t.wh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/input.ll -o %t.wh.O2.s
; RUN: %python %S/Inputs/check-tensix-contents-record-only.py --assembly %t.wh.O2.s --mir %t.wh.O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/input.ll -o %t.wh.O2.o
;
; Recording has no numerical result. The first store still observes %old;
; only each real execute produces a new SSA value. This tests direct LLVM
; receivers; source Task/Buffer producers and hardware are separate gates.
;
; VREG-LABEL: name:{{ *}}record_only
; VREG: [[OLD:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; VREG: [[CAP:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; VREG: PseudoTTContentsReplayTemplateBegin 41, 0, 0, 0, 1, [[OLD]], [[CAP]]
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 41, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateEnd 41, [[OLD]],
; VREG: TTSFPSTORE [[OLD]], 0, 0, 4,
; VREG: early-clobber [[FIRST:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute1 41, 1, [[OLD]], [[CAP]],
; VREG: TTSFPSTORE [[FIRST]], 8, 0, 4,
; VREG: early-clobber [[SECOND:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute1 41, 1, [[FIRST]], {{(killed )?}}[[CAP]],
; VREG: TTSFPSTORE {{(killed )?}}[[SECOND]], 16, 0, 4,
; VREG-LABEL: name:{{ *}}record_only_loop
; VREG: PseudoTTContentsReplayTemplateBegin 42, 0, 0, 0,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 42, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateEnd 42,
; VREG: {{%[0-9]+}}:sfpr = PHI
; VREG: early-clobber {{%[0-9]+}}:sfpr = PseudoTTContentsReplayTemplateExecute1 42,
; VREG-LABEL: name:{{ *}}record_all_recipes
; VREG: PseudoTTContentsReplayTemplateBegin 43, 0, 0, 0,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 43, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateRecordAnd 43, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateRecordOr 43, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateRecordXor 43, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateRecordNot 43, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateEnd 43,
; VREG-LABEL: name:{{ *}}record_only_unused
; VREG: PseudoTTContentsReplayTemplateBegin 44, 0, 0, 0,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 44, 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; VREG: PseudoTTContentsReplayTemplateEnd 44,
;
; PHYS-LABEL: name:{{ *}}record_only
; PHYS: PseudoTTContentsSFPUReplay 1, 0, 1, 0,
; PHYS: PseudoTTSFPURecordWord {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue{{$}}
; PHYS: PseudoTTReplayRecordEnd
; PHYS: TTSFPSTORE
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 0,
; PHYS: TTSFPSTORE
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 0,
; PHYS: TTSFPSTORE
; PHYS-LABEL: name:{{ *}}record_only_loop
; PHYS: PseudoTTContentsSFPUReplay 1, 0, 1, 0,
; PHYS: PseudoTTSFPURecordWord
; PHYS: PseudoTTReplayRecordEnd
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 0,
; PHYS: {{BNE|BEQ|BLT|BGE|BLTU|BGEU|PseudoBR}} {{.*}}%bb.
; PHYS-LABEL: name:{{ *}}record_all_recipes
; PHYS: PseudoTTContentsSFPUReplay 1, 0,
; PHYS-LABEL: name:{{ *}}record_only_unused
; PHYS: PseudoTTContentsSFPUReplay 1, 0, 1, 0,
; PHYS-NOT: PseudoTTContentsSFPUReplay 0, 0,
;
; VREG-LABEL: name:{{ *}}two_records
; VREG: PseudoTTContentsReplayTemplateBegin 51, 0, 0, 0,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 51, 0, 1,
; VREG: PseudoTTContentsReplayTemplateBegin 52, 0, 0, 0,
; VREG: PseudoTTContentsReplayTemplateRecordIAdd 52, 0, 1,
; VREG: PseudoTTContentsReplayTemplateExecute1 51,
; VREG: PseudoTTContentsReplayTemplateExecute1 52,
; PHYS-LABEL: name:{{ *}}two_records
; PHYS: PseudoTTContentsSFPUReplay 1, 0, 1, 0,
; PHYS: PseudoTTContentsSFPUReplay 1, 0, 1, 1,
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 0,
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 1,
;
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
declare void @llvm.riscv.tt.contents.replay.template.record.and(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.or(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.xor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.not(i32 immarg, i32 immarg)

;--- positive.ll
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
  %second.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 16, i32 0, i32 4)
  ret void
}

define void @record_only_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 42, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 42, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 42, <32 x i32> %old)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %value = phi <32 x i32> [ %old, %entry ], [ %next, %loop ]
  %index = phi i32 [ 0, %entry ], [ %step, %loop ]
  %next.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 42, i32 1, <32 x i32> %value, <32 x i32> %capture)
  %next = extractvalue { <32 x i32> } %next.replay, 0
  %step = add i32 %index, 1
  %done = icmp eq i32 %step, %count
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [ %old, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 0, i32 0, i32 4)
  ret void
}

define void @record_all_recipes() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 43, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 43, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.and(i32 43, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.or(i32 43, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 43, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.not(i32 43, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 43, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %first.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 43, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 8, i32 0, i32 4)
  ret void
}

define void @record_only_unused() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 44, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 44, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 44, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  ret void
}

define void @two_records() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 51, i32 0, i32 0, i32 0, i32 1, <32 x i32> %a, <32 x i32> %b)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 51, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 51, <32 x i32> %a)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 52, i32 0, i32 0, i32 0, i32 1, <32 x i32> %c, <32 x i32> %d)
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

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
