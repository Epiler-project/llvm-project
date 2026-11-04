; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/good.ll -o %t/bh-O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/good.ll -o %t/bh-O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/good.ll -o %t/wh-O0.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/good.ll -o %t/wh-O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/good.ll -o - | FileCheck %s --check-prefix=BODY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/good.ll -o - | FileCheck %s --check-prefix=BODY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -filetype=obj %t/bad-iadd-mode.ll -o %t/bad-iadd-mode.o 2>&1 | FileCheck %s --check-prefix=IADD-MODE
; RUN: test ! -e %t/bad-iadd-mode.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %t/bad-loadi-mode.ll -o %t/bad-loadi-mode.o 2>&1 | FileCheck %s --check-prefix=LOADI-MODE
; RUN: test ! -e %t/bad-loadi-mode.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %t/bad-loadi-value.ll -o %t/bad-loadi-value.o 2>&1 | FileCheck %s --check-prefix=LOADI-VALUE
; RUN: test ! -e %t/bad-loadi-value.o

; Record-and-execute must preserve each SSA old value, both read-only captures,
; native immediates, and actual arithmetic instructions in the recorded body.
; BODY-LABEL: name: direct_arithmetic
; BODY: PseudoTTContentsSFPUReplay 1, 1,
; BODY: TTSFPADD
; BODY: TTSFPMUL
; BODY: TTSFPMAD
; BODY: TTSFPISUB
; BODY: TTSFPMOVNeg
; BODY: TTSFPLOADI
; BODY: PseudoTTContentsSFPUReplay 0, 0,
; BODY-LABEL: name: direct_condition_updates
; BODY: PseudoTTContentsSFPUReplay 1, 1,
; BODY: TTSFPISUBCCLT
; BODY: TTSFPISUBCCGE
; BODY: PseudoTTContentsSFPUReplay 0, 0,

;--- good.ll
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
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @direct_arithmetic() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %a, <32 x i32> %b)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %old, <32 x i32> %a, <32 x i32> %b, i32 0)
  %mul = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %add, <32 x i32> %add, <32 x i32> %b, i32 0)
  %mad = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %mul, <32 x i32> %mul, <32 x i32> %b, <32 x i32> %a, i32 0)
  %sub = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %mad, <32 x i32> %a, i32 6)
  %neg = call <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32> %sub, <32 x i32> %sub, i32 1)
  ; Upper-half update keeps the arithmetic chain observable in the lower half.
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %neg, i32 4096, i32 8)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %loaded)
  %executed = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %loaded, <32 x i32> %a, <32 x i32> %b)
  %result = extractvalue { <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

define void @direct_condition_updates() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 72, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %source)
  %lt = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %source, i32 2)
  %ge = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %lt, <32 x i32> %source, i32 10)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 72, <32 x i32> %ge)
  %executed = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 72, i32 1, <32 x i32> %ge, <32 x i32> %source)
  %result = extractvalue { <32 x i32> } %executed, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 4)
  ret void
}

; IADD-MODE: unsupported contents instruction form or mode
;--- bad-iadd-mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
define void @bad_iadd_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %source)
  %invalid = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %source, i32 3)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %invalid)
  ret void
}

; LOADI-MODE: SFPLOADI mode must be one of
;--- bad-loadi-mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
define void @bad_loadi_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %invalid = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 7, i32 7)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %invalid)
  ret void
}

; LOADI-VALUE: Tensix immediate operand must fit in 16 unsigned bits
;--- bad-loadi-value.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
define void @bad_loadi_value() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %invalid = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65536, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %invalid)
  ret void
}
