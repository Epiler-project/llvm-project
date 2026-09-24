; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/preserve.ll -o - | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/preserve.ll -o - | FileCheck %s --check-prefix=PRESERVE
; RUN: cat %t/common.ll %t/external.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/claimed-readnone.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/indirect.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/interposable.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/writing.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/nested-writing.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/inline-asm.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/target-state.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: cat %t/common.ll %t/atomic.ll | not llc -mtriple=riscv32 -mattr=+a,+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; REJECT: Tensix MOP execution requires all seven instruction slots to be configured

; Scalar polling does not alter the MOP template or a completed recording.
; Keep the calls and polling backedges: do not inline, duplicate or reprogram
; the template to work around missing preservation analysis.
; PRESERVE-LABEL: poll:
; PRESERVE: fence
; PRESERVE: lw
; PRESERVE: bnez
; PRESERVE-LABEL: poll_nested:
; PRESERVE: {{call|tail}} poll
; PRESERVE-LABEL: preserve:
; PRESERVE: .word 0x10000044
; PRESERVE: .word 0x08000000
; PRESERVE: call poll_nested
; PRESERVE: .word 0x06000000
; PRESERVE: ret

;--- preserve.ll
declare void @llvm.riscv.tt.nop.mop(i32 immarg)
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()

define internal void @poll(ptr %address) noinline {
entry:
  br label %loop
loop:
  fence seq_cst
  %value = load volatile i32, ptr %address, align 4
  %busy = and i32 %value, 254
  %continue = icmp ne i32 %busy, 0
  br i1 %continue, label %loop, label %exit
exit:
  fence seq_cst
  ret void
}

define internal void @poll_nested(ptr %address) noinline {
  call void @poll(ptr %address)
  ret void
}

define void @preserve(ptr %address) "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.nop.mop(i32 3)
  call void @llvm.riscv.tt.nop.mop(i32 4)
  call void @llvm.riscv.tt.nop.mop(i32 5)
  call void @llvm.riscv.tt.nop.mop(i32 6)
  call void @llvm.riscv.tt.nop.mop(i32 7)
  call void @llvm.riscv.tt.replay.mop(i32 8, i32 0, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @poll_nested(ptr %address)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}

;--- common.ll
declare void @llvm.riscv.tt.nop.mop(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)

define void @reject(ptr %address, ptr %callee) "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.nop.mop(i32 3)
  call void @llvm.riscv.tt.nop.mop(i32 4)
  call void @llvm.riscv.tt.nop.mop(i32 5)
  call void @llvm.riscv.tt.nop.mop(i32 6)
  call void @llvm.riscv.tt.nop.mop(i32 7)
  call void @llvm.riscv.tt.nop.mop(i32 8)
  call void @change(ptr %address, ptr %callee)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}

;--- external.ll
declare void @change(ptr, ptr)

;--- claimed-readnone.ll
; Memory effects alone do not describe special target state.
declare void @change(ptr, ptr) memory(none)

;--- indirect.ll
define internal void @change(ptr %address, ptr %callee) noinline {
  call void %callee(ptr %address)
  ret void
}

;--- interposable.ll
; A visible load-only body is not evidence for the actual weak definition.
define weak void @change(ptr %address, ptr %callee) noinline {
  %value = load volatile i32, ptr %address, align 4
  ret void
}

;--- writing.ll
define internal void @change(ptr %address, ptr %callee) noinline {
  store volatile i32 0, ptr %address, align 4
  ret void
}

;--- nested-writing.ll
define internal void @write(ptr %address) noinline {
  store volatile i32 0, ptr %address, align 4
  ret void
}
define internal void @change(ptr %address, ptr %callee) noinline {
  call void @write(ptr %address)
  ret void
}

;--- inline-asm.ll
define internal void @change(ptr %address, ptr %callee) noinline {
  call void asm sideeffect "", ""()
  ret void
}

;--- target-state.ll
define internal void @change(ptr %address, ptr %callee) noinline "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.nop.mop(i32 2)
  ret void
}

;--- atomic.ll
define internal void @change(ptr %address, ptr %callee) noinline {
  %old = atomicrmw add ptr %address, i32 1 seq_cst
  ret void
}
