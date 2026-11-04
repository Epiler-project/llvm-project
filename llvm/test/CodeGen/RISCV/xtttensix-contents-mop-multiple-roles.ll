; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/valid.ll > %t/valid-module.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/valid-module.ll -o - | FileCheck %s --check-prefix=RESULT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid-module.ll -o %t/bh.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid-module.ll -o %t/wh.o
; RUN: cat %t/common.ll %t/invalid-role.ll > %t/invalid-role-module.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/invalid-role-module.ll -o %t/invalid-role.o 2>&1 | FileCheck %s --check-prefix=BAD-ROLE
; RUN: not test -e %t/invalid-role.o
; RUN: cat %t/common.ll %t/invalid-arity.ll > %t/invalid-arity-module.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/invalid-arity-module.ll -o %t/invalid-arity.o 2>&1 | FileCheck %s --check-prefix=BAD-ARITY
; RUN: not test -e %t/invalid-arity.o
; RUN: cat %t/common.ll %t/invalid-capture.ll > %t/invalid-capture-module.ll
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/invalid-capture-module.ll -o %t/invalid-capture.o 2>&1 | FileCheck %s --check-prefix=BAD-CAPTURE
; RUN: not test -e %t/invalid-capture.o
;
; MOP executes a recorded template with flat roles. Each invocation returns
; exactly one new SSA contents value per mutable; read-only captures remain
; unchanged, and a second invocation consumes the first invocation's results.
; The existing one-mutable/one-capture MOP test remains independent.
; RESULT-LABEL: name:{{ *}}mop_two_mutable
; RESULT: TTMOP 0, 0, 1,{{.*}}implicit-def $tt_l{{[0-9]+}}{{.*}}implicit-def $tt_l{{[0-9]+}}
; RESULT: TTSFPSTORE
; RESULT: TTSFPSTORE
; RESULT: TTMOP 0, 0, 1,{{.*}}implicit-def $tt_l{{[0-9]+}}{{.*}}implicit-def $tt_l{{[0-9]+}}
; RESULT: TTSFPSTORE
; RESULT: TTSFPSTORE
; RESULT-LABEL: name:{{ *}}mop_two_captures
; RESULT: TTMOP 0, 0, 1,
; RESULT: TTSFPSTORE
; RESULT: TTMOP 0, 0, 1,
; RESULT: TTSFPSTORE
; BAD-ROLE: contents replay recipe source requires a declared contents role
; BAD-ARITY: contents replay execute arity differs from its recorded roles
; BAD-CAPTURE: contents replay execute requires its recorded read-only capture

;--- common.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.record.xor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)

;--- valid.ll
define void @mop_two_mutable() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 2, <32 x i32> %a, <32 x i32> %b, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 71, i32 1, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %a, <32 x i32> %b)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 71, i32 2, <32 x i32> %a, <32 x i32> %b, <32 x i32> %capture)
  %a.first = extractvalue { <32 x i32>, <32 x i32> } %first, 0
  %b.first = extractvalue { <32 x i32>, <32 x i32> } %first, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %a.first, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %b.first, i32 8, i32 0, i32 4)
  %second = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 71, i32 2, <32 x i32> %a.first, <32 x i32> %b.first, <32 x i32> %capture)
  %a.second = extractvalue { <32 x i32>, <32 x i32> } %second, 0
  %b.second = extractvalue { <32 x i32>, <32 x i32> } %second, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %a.second, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %b.second, i32 24, i32 0, i32 4)
  ret void
}

define void @mop_two_captures() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 72, i32 0, i32 0, i32 0, i32 1, <32 x i32> %a, <32 x i32> %capture0, <32 x i32> %capture1)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 72, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.replay.template.record.xor(i32 72, i32 0, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 72, <32 x i32> %a)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 72, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 72, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 72, i32 1, <32 x i32> %a, <32 x i32> %capture0, <32 x i32> %capture1)
  %a.first = extractvalue { <32 x i32> } %first, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %a.first, i32 0, i32 0, i32 4)
  %second = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 72, i32 1, <32 x i32> %a.first, <32 x i32> %capture0, <32 x i32> %capture1)
  %a.second = extractvalue { <32 x i32> } %second, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %a.second, i32 8, i32 0, i32 4)
  ret void
}

;--- invalid-role.ll
define void @invalid_role() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 73, i32 0, i32 0, i32 0, i32 2, <32 x i32> %a, <32 x i32> %b, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 73, i32 0, i32 3)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 73, <32 x i32> %a, <32 x i32> %b)
  ret void
}

;--- invalid-arity.ll
define void @invalid_arity() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 74, i32 0, i32 0, i32 0, i32 2, <32 x i32> %a, <32 x i32> %b, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 74, i32 0, i32 2)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 74, <32 x i32> %a, <32 x i32> %b)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 74, i32 5)
  %pair = call { <32 x i32>, <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32v32i32s(i32 74, i32 2, <32 x i32> %a, <32 x i32> %b)
  %next = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 0, i32 0, i32 4)
  ret void
}

;--- invalid-capture.ll
define void @invalid_capture() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %capture1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 75, i32 0, i32 0, i32 0, i32 1, <32 x i32> %a, <32 x i32> %capture0, <32 x i32> %capture1)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 75, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 75, <32 x i32> %a)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 75, i32 5)
  %result = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 75, i32 1, <32 x i32> %a, <32 x i32> %capture1, <32 x i32> %capture0)
  %next = extractvalue { <32 x i32> } %result, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %next, i32 0, i32 0, i32 4)
  ret void
}
