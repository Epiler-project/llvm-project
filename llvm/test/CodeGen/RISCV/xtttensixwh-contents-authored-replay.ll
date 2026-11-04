; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/positive.ll > %t/input.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/input.ll -o - | FileCheck %s --check-prefix=VREG
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/input.ll -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O2.o
; RUN: llvm-objdump -d --mattr=+xtttensixwh %t/O0.o | FileCheck %s --check-prefix=OBJ
; RUN: llvm-objdump -d --mattr=+xtttensixwh %t/O2.o | FileCheck %s --check-prefix=OBJ
; RUN: cat %t/common.ll %t/record-only.ll > %t/bad-mode.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/bad-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: cat %t/common.ll %t/wrong-capture.ll > %t/bad-capture.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/bad-capture.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAPTURE
;
; The first body really executes once while recording; execute applies the
; same instruction to the current mutable contents and returns its new value.
; C15 is snapshotted into one read-only contents capture, live through execute.
; Missing execute def/use or capture lifetime must break the MIR checks.
; This is direct LLVM receiver coverage, not a source producer or device test.
;
; VREG-LABEL: name:{{ *}}contents_authored_replay
; VREG: [[OLD:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; VREG: [[CAP:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; VREG: PseudoTTContentsReplayTemplateBegin 41, 1, 0, 0, 1, [[OLD]], [[CAP]]
; VREG: [[FIRST:%[0-9]+]]:sfpr = TTSFPIADD [[OLD]], [[CAP]]
; VREG: PseudoTTContentsReplayTemplateEnd 41, [[FIRST]]
; VREG: early-clobber [[SECOND:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute1 41, 1, [[FIRST]], {{(killed )?}}[[CAP]]
; VREG: TTSFPSTORE {{(killed )?}}[[SECOND]], 8, 0, 4
; PHYS-LABEL: name:{{ *}}contents_authored_replay
; PHYS: PseudoTTContentsSFPUReplay 1, 1, 1, 0
; PHYS: [[DST:\$tt_l[0-7]]] = TTSFPIADD [[DST]], [[CAP:\$tt_l[0-7]]]
; PHYS: TTSFPSTORE{{.*}}[[DST]], 0, 0, 4
; PHYS: PseudoTTContentsSFPUReplay 0, 0, 1, 0,
; PHYS-SAME: implicit [[DST]]
; PHYS-SAME: implicit [[CAP]]
; PHYS-SAME: implicit-def [[DST]]
; PHYS: TTSFPSTORE{{.*}}[[DST]], 8, 0, 4
; OBJ-LABEL: <contents_authored_replay>:
; OBJ: 1000004c
; OBJ: e400{{[0-9a-f][0-9a-f][0-9a-f][0-9a-f]}} ttsfpiadd
; OBJ: 10000040
; OBJ: ret
; MODE: contents record-only body requires a non-executing recipe
; CAPTURE: contents replay execute requires its recorded read-only capture

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

;--- positive.ll
define void @contents_authored_replay() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 0, i32 0, i32 4)
  %second.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.replay, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 8, i32 0, i32 4)
  ret void
}

;--- record-only.ll
define void @record_only() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  ret void
}

;--- wrong-capture.ll
define void @wrong_capture() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 41, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %capture, i32 4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 41, <32 x i32> %first)
  %second.replay = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 41, i32 1, <32 x i32> %first, <32 x i32> %old)
  %second = extractvalue { <32 x i32> } %second.replay, 0
  ret void
}
