; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.o0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.o2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh.o0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.o2.o
;
; A single record-and-execute template has two mutable result chains and two
; read-only captures. Its multi-step arithmetic body is executed while recording
; and later as an MOP. The source loop stays a CFG loop around MOP execution.
;
; VREG-LABEL: name: mop_record_execute_multi_role_loop
; VREG-DAG: [[MUTABLE_A:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; VREG-DAG: [[MUTABLE_B:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; VREG-DAG: [[CAPTURE:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; VREG-DAG: [[CAPTURE2:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c8
; VREG: PseudoTTContentsReplayTemplateBegin 91, 1, 0, 0, 2, [[MUTABLE_A]], [[MUTABLE_B]], [[CAPTURE]], [[CAPTURE2]],
; VREG: TTSFPADD
; VREG: TTSFPMUL
; VREG: TTSFPISUB
; VREG: TTSFPMAD
; VREG: TTSFPMOVNeg
; VREG: TTSFPLOADI
; VREG: PseudoTTContentsReplayTemplateEnd 91, [[INIT_A:%[0-9]+]], [[INIT_B:%[0-9]+]],
; VREG: [[A:%[0-9]+]]:sfpr = PHI [[INIT_A]], %bb.{{[0-9]+}}, [[NEXTA:%[0-9]+]], %bb.{{[0-9]+}}
; VREG: [[B:%[0-9]+]]:sfpr = PHI [[INIT_B]], %bb.{{[0-9]+}}, [[NEXTB:%[0-9]+]], %bb.{{[0-9]+}}
; VREG: early-clobber [[NEXTA]]:sfpr, early-clobber [[NEXTB]]:sfpr = PseudoTTContentsReplayTemplateMopExecute2 91, 2, [[A]], [[B]], [[CAPTURE]], [[CAPTURE2]],
; VREG-DAG: {{(br|PseudoBR)}} {{.*}}%bb.
; VREG-DAG: {{(BNE|BEQ|BLT|BGE|BLTU|BGEU)}} {{.*}}%bb.
;
; The selected MOP has two distinct results tied to their corresponding old
; mutable values; both captures are uses, not extra results or mutables.
; PHYS-LABEL: name: mop_record_execute_multi_role_loop
; PHYS: [[A:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c9
; PHYS: [[B:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c10
; PHYS: [[C0:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c15
; PHYS: [[C1:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c8
; PHYS: PseudoTTContentsSFPUReplay 1, 1,
; PHYS: TTMOP 0, 0, 1,{{.*}}implicit [[A]], implicit [[B]], implicit [[C0]], implicit [[C1]], implicit-def [[A]], implicit-def [[B]]
; PHYS: {{BNE|BEQ|BLT|BGE|BLTU|BGEU|PseudoBR}} {{.*}}%bb.

target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)

define void @mop_record_execute_multi_role_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %capture2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 91, i32 1, i32 0, i32 0, i32 2, <32 x i32> %a0, <32 x i32> %b0, <32 x i32> %capture, <32 x i32> %capture2)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %a0, <32 x i32> %b0, <32 x i32> %capture2, i32 0)
  %b1 = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %b0, <32 x i32> %a1, <32 x i32> %capture, i32 0)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %a1, <32 x i32> %b1, i32 6)
  %b2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %b1, <32 x i32> %a2, <32 x i32> %capture, <32 x i32> %b1, i32 0)
  ; MOVNeg consumes the current A chain; LOADI keeps B's lower half.
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32> %a2, <32 x i32> %a2, i32 1)
  %b3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %b2, i32 4096, i32 8)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 91, <32 x i32> %a3, <32 x i32> %b3)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 91, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 91, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %empty = icmp eq i32 %limit, 0
  br i1 %empty, label %exit, label %loop

loop:
  %iteration = phi i32 [ 0, %entry ], [ %next.iteration, %latch ]
  %a = phi <32 x i32> [ %a3, %entry ], [ %a.next, %latch ]
  %b = phi <32 x i32> [ %b3, %entry ], [ %b.next, %latch ]
  %updated = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 91, i32 2, <32 x i32> %a, <32 x i32> %b, <32 x i32> %capture, <32 x i32> %capture2)
  %a.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %b.next = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  br label %latch

latch:
  %next.iteration = add i32 %iteration, 1
  %done = icmp eq i32 %next.iteration, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  %a.final = phi <32 x i32> [ %a3, %entry ], [ %a.next, %latch ]
  %b.final = phi <32 x i32> [ %b3, %entry ], [ %b.next, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %a.final, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %b.final, i32 8, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
