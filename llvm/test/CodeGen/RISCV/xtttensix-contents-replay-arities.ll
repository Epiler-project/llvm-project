; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh.O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.O2.o
;
; Every fixed result arity must produce real SFPR definitions at ISel.
; Each mutable changes in the recorded native body, the literal result
; struct is unpacked, and every new SSA value is stored. With no captures,
; the eight-result case reaches all eight available mutable register roles.
; No source loop or loop body is generated or duplicated.
;
; ISEL-LABEL: name:{{ *}}execute_3_mutables
; ISEL: [[M3OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M3OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M3OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M3NEW0:%[0-9]+]]:sfpr, early-clobber [[M3NEW1:%[0-9]+]]:sfpr, early-clobber [[M3NEW2:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute3 103, 3, [[M3OLD0]], [[M3OLD1]], [[M3OLD2]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M3NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M3NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M3NEW2]], 16, 0, 4,
;
; ISEL-LABEL: name:{{ *}}execute_4_mutables
; ISEL: [[M4OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M4OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M4OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M4OLD3:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M4NEW0:%[0-9]+]]:sfpr, early-clobber [[M4NEW1:%[0-9]+]]:sfpr, early-clobber [[M4NEW2:%[0-9]+]]:sfpr, early-clobber [[M4NEW3:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute4 104, 4, [[M4OLD0]], [[M4OLD1]], [[M4OLD2]], [[M4OLD3]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M4NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M4NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M4NEW2]], 16, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M4NEW3]], 24, 0, 4,
;
; ISEL-LABEL: name:{{ *}}execute_5_mutables
; ISEL: [[M5OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M5OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M5OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M5OLD3:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M5OLD4:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M5NEW0:%[0-9]+]]:sfpr, early-clobber [[M5NEW1:%[0-9]+]]:sfpr, early-clobber [[M5NEW2:%[0-9]+]]:sfpr, early-clobber [[M5NEW3:%[0-9]+]]:sfpr, early-clobber [[M5NEW4:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute5 105, 5, [[M5OLD0]], [[M5OLD1]], [[M5OLD2]], [[M5OLD3]], [[M5OLD4]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M5NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M5NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M5NEW2]], 16, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M5NEW3]], 24, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M5NEW4]], 32, 0, 4,
;
; ISEL-LABEL: name:{{ *}}execute_6_mutables
; ISEL: [[M6OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M6OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M6OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M6OLD3:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M6OLD4:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M6OLD5:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M6NEW0:%[0-9]+]]:sfpr, early-clobber [[M6NEW1:%[0-9]+]]:sfpr, early-clobber [[M6NEW2:%[0-9]+]]:sfpr, early-clobber [[M6NEW3:%[0-9]+]]:sfpr, early-clobber [[M6NEW4:%[0-9]+]]:sfpr, early-clobber [[M6NEW5:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute6 106, 6, [[M6OLD0]], [[M6OLD1]], [[M6OLD2]], [[M6OLD3]], [[M6OLD4]], [[M6OLD5]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW2]], 16, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW3]], 24, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW4]], 32, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M6NEW5]], 40, 0, 4,
;
; ISEL-LABEL: name:{{ *}}execute_7_mutables
; ISEL: [[M7OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD3:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD4:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD5:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M7OLD6:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M7NEW0:%[0-9]+]]:sfpr, early-clobber [[M7NEW1:%[0-9]+]]:sfpr, early-clobber [[M7NEW2:%[0-9]+]]:sfpr, early-clobber [[M7NEW3:%[0-9]+]]:sfpr, early-clobber [[M7NEW4:%[0-9]+]]:sfpr, early-clobber [[M7NEW5:%[0-9]+]]:sfpr, early-clobber [[M7NEW6:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute7 107, 7, [[M7OLD0]], [[M7OLD1]], [[M7OLD2]], [[M7OLD3]], [[M7OLD4]], [[M7OLD5]], [[M7OLD6]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW2]], 16, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW3]], 24, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW4]], 32, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW5]], 40, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M7NEW6]], 48, 0, 4,
;
; ISEL-LABEL: name:{{ *}}execute_8_mutables
; ISEL: [[M8OLD0:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD1:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD2:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD3:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD4:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD5:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD6:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: [[M8OLD7:%[0-9]+]]:sfpr = TTSFPNOT
; ISEL: early-clobber [[M8NEW0:%[0-9]+]]:sfpr, early-clobber [[M8NEW1:%[0-9]+]]:sfpr, early-clobber [[M8NEW2:%[0-9]+]]:sfpr, early-clobber [[M8NEW3:%[0-9]+]]:sfpr, early-clobber [[M8NEW4:%[0-9]+]]:sfpr, early-clobber [[M8NEW5:%[0-9]+]]:sfpr, early-clobber [[M8NEW6:%[0-9]+]]:sfpr, early-clobber [[M8NEW7:%[0-9]+]]:sfpr = PseudoTTContentsReplayTemplateExecute8 108, 8, [[M8OLD0]], [[M8OLD1]], [[M8OLD2]], [[M8OLD3]], [[M8OLD4]], [[M8OLD5]], [[M8OLD6]], [[M8OLD7]],
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW0]], 0, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW1]], 8, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW2]], 16, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW3]], 24, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW4]], 32, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW5]], 40, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW6]], 48, 0, 4,
; ISEL: TTSFPSTORE {{(killed )?}}[[M8NEW7]], 56, 0, 4,
;
target triple = "riscv32"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32v32i32v32i32s(i32 immarg, i32 immarg, ...)

define void @execute_3_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 103, i32 1, i32 0, i32 0, i32 3, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 103, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32s(i32 103, i32 3, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  ret void
}

define void @execute_4_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  %old3.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old3.seed, i32 4, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 104, i32 1, i32 0, i32 0, i32 4, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  %recorded3 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old3)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 104, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32s(i32 104, i32 4, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  %next3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next3, i32 24, i32 0, i32 4)
  ret void
}

define void @execute_5_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  %old3.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old3.seed, i32 4, i32 2)
  %old4.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old4.seed, i32 5, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 105, i32 1, i32 0, i32 0, i32 5, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  %recorded3 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old3)
  %recorded4 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old4)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 105, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32s(i32 105, i32 5, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  %next3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next3, i32 24, i32 0, i32 4)
  %next4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next4, i32 32, i32 0, i32 4)
  ret void
}

define void @execute_6_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  %old3.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old3.seed, i32 4, i32 2)
  %old4.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old4.seed, i32 5, i32 2)
  %old5.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old5.seed, i32 6, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 106, i32 1, i32 0, i32 0, i32 6, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  %recorded3 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old3)
  %recorded4 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old4)
  %recorded5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old5)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 106, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32s(i32 106, i32 6, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  %next3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next3, i32 24, i32 0, i32 4)
  %next4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next4, i32 32, i32 0, i32 4)
  %next5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next5, i32 40, i32 0, i32 4)
  ret void
}

define void @execute_7_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  %old3.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old3.seed, i32 4, i32 2)
  %old4.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old4.seed, i32 5, i32 2)
  %old5.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old5.seed, i32 6, i32 2)
  %old6.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old6 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old6.seed, i32 7, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 107, i32 1, i32 0, i32 0, i32 7, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5, <32 x i32> %old6)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  %recorded3 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old3)
  %recorded4 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old4)
  %recorded5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old5)
  %recorded6 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old6)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 107, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5, <32 x i32> %recorded6)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32v32i32s(i32 107, i32 7, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5, <32 x i32> %recorded6)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  %next3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next3, i32 24, i32 0, i32 4)
  %next4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next4, i32 32, i32 0, i32 4)
  %next5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next5, i32 40, i32 0, i32 4)
  %next6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 6
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next6, i32 48, i32 0, i32 4)
  ret void
}

define void @execute_8_mutables() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old0.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old0.seed, i32 1, i32 2)
  %old1.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old1.seed, i32 2, i32 2)
  %old2.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old2.seed, i32 3, i32 2)
  %old3.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old3.seed, i32 4, i32 2)
  %old4.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old4.seed, i32 5, i32 2)
  %old5.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old5.seed, i32 6, i32 2)
  %old6.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old6 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old6.seed, i32 7, i32 2)
  %old7.seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old7 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old7.seed, i32 8, i32 2)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 108, i32 1, i32 0, i32 0, i32 8, <32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5, <32 x i32> %old6, <32 x i32> %old7)
  %recorded0 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old0)
  %recorded1 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old1)
  %recorded2 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old2)
  %recorded3 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old3)
  %recorded4 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old4)
  %recorded5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old5)
  %recorded6 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old6)
  %recorded7 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %old7)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 108, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5, <32 x i32> %recorded6, <32 x i32> %recorded7)
  %executed = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32v32i32v32i32v32i32v32i32v32i32v32i32v32i32s(i32 108, i32 8, <32 x i32> %recorded0, <32 x i32> %recorded1, <32 x i32> %recorded2, <32 x i32> %recorded3, <32 x i32> %recorded4, <32 x i32> %recorded5, <32 x i32> %recorded6, <32 x i32> %recorded7)
  %next0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next0, i32 0, i32 0, i32 4)
  %next1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next1, i32 8, i32 0, i32 4)
  %next2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next2, i32 16, i32 0, i32 4)
  %next3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next3, i32 24, i32 0, i32 4)
  %next4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next4, i32 32, i32 0, i32 4)
  %next5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next5, i32 40, i32 0, i32 4)
  %next6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 6
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next6, i32 48, i32 0, i32 4)
  %next7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %executed, 7
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next7, i32 56, i32 0, i32 4)
  ret void
}
