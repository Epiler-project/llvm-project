; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/mop-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/remote-instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/pcbuf.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/expander-donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/backend-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/outside-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/wrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/mop-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/remote-instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/pcbuf.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/expander-donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/backend-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/outside-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/wrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/mop-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/remote-instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/pcbuf.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/expander-donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/backend-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/outside-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/wrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/mop-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/remote-instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/pcbuf.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/expander-donecheck.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/backend-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/outside-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/wrap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/bh-end.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/bh-end.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/cross-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/bh-end.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/bh-end.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/cross-l1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WRITE

; Precise local L1 and compiler-owned stack accesses cannot reprogram MOP.
; Unknown, MMIO, instruction aliases and crossing ranges remain rejected.
; Generic DoneCheck stores do not stand in for the typed completion owner.
; WRITE: contents MOP profile cannot admit unknown scalar writes

;--- valid.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @local_l1(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  ; The typed address reset touches thread config and Dst counters, not MOP cells.
  call void @llvm.riscv.tt.contents.macro.address.reset()
  store volatile i32 %value, ptr inttoptr (i32 1048832 to ptr), align 4
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

define void @response(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.read.response(i32 %value)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}



define void @wait_live_contents(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.coprocessor.wait()
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

define void @l1_common_end(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 1499132 to ptr), align 4
  ret void
}

;--- bh-end.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @blackhole_l1_end(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 1572860 to ptr), align 4
  ret void
}

;--- unknown.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @unknown(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr %unknown, align 1
  ret void
}

;--- mop-config.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @mop_config(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -4718592 to ptr), align 1
  ret void
}

;--- instruction-port.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @instruction_port(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1835008 to ptr), align 1
  ret void
}

;--- remote-instruction-port.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @remote_instruction_port(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1769472 to ptr), align 1
  ret void
}

;--- pcbuf.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @pcbuf(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1572864 to ptr), align 1
  ret void
}

;--- donecheck.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @donecheck(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1572860 to ptr), align 1
  ret void
}

;--- expander-donecheck.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @expander_donecheck(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1572856 to ptr), align 1
  ret void
}

;--- backend-config.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @backend_config(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 -1114112 to ptr), align 1
  ret void
}

;--- outside-l1.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @outside_l1(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr inttoptr (i32 1572864 to ptr), align 1
  ret void
}

;--- wrap.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @wrap(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i32 %value, ptr getelementptr (i8, ptr inttoptr (i32 -4 to ptr), i64 1), align 1
  ret void
}

;--- cross-l1.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.riscv.tt.coprocessor.wait()
define void @cross_l1(ptr %unknown, i32 %value) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  store volatile i64 42, ptr inttoptr (i32 1499132 to ptr), align 1
  ret void
}
