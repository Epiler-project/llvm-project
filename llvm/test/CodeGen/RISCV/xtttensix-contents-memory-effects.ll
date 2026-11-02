; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unknown-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unknown-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unknown-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unknown-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unknown-scalar-address.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unknown-scalar-address.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/instruction-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/instruction-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/instruction-alias-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/instruction-alias-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/remote-instruction-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/remote-instruction-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/donecheck-overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/donecheck-overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/l1-overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/l1-overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/wallclock-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/wallclock-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/scalar-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/scalar-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/inline-asm.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/inline-asm.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG

; Scalar MMIO completion and readback do not issue SFPCONFIG. Preserve the
; reaching LaneConfig only for a complete, fixed-width access within a known
; hardware owner. Volatile loads are ordered effects in LLVM and need the same
; classification as stores. Unknown addresses, instruction-window aliases,
; accesses crossing a range boundary, and opaque calls remain conservative.
; CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled

;--- valid.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @manual_ttsync() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr inttoptr (i32 -1572860 to ptr), align 4
  %done = load volatile i32, ptr inttoptr (i32 -1572860 to ptr), align 4
  call void @llvm.riscv.tt.read.response(i32 %done)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

define void @donecheck_load_only() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %done = load volatile i32, ptr inttoptr (i32 -1572856 to ptr), align 4
  call void @llvm.riscv.tt.read.response(i32 %done)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

; ConstantHoisting materializes address integers as same-type bitcast SSA.
; This must remain a known hardware address at both optimization levels.
define void @hoisted_donecheck() "tensix-executor"="trisc1" {
  %address = bitcast i32 -1572860 to i32
  %pointer = inttoptr i32 %address to ptr
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr %pointer, align 4
  %done = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %done)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

define void @l1_readback() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %output = getelementptr i32, ptr inttoptr (i32 1048576 to ptr), i32 3
  store volatile i32 0, ptr %output, align 4
  %observed = load volatile i32, ptr %output, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

define void @clock_reads() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %low = load volatile i32, ptr inttoptr (i32 -5168656 to ptr), align 4
  %high = load volatile i32, ptr inttoptr (i32 -5168652 to ptr), align 4
  %latched = load volatile i32, ptr inttoptr (i32 -5168648 to ptr), align 4
  call void @llvm.riscv.tt.read.response(i32 %low)
  call void @llvm.riscv.tt.read.response(i32 %high)
  call void @llvm.riscv.tt.read.response(i32 %latched)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- unknown-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @unknown_store(ptr %address) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr %address, align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- unknown-load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @unknown_load(ptr %address) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %observed = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- unknown-scalar-address.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @unknown_scalar_address(i32 %unknown) "tensix-executor"="trisc1" {
  %address = bitcast i32 %unknown to i32
  %pointer = inttoptr i32 %address to ptr
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr %pointer, align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- instruction-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @instruction_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr inttoptr (i32 4293132288 to ptr), align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- instruction-alias-load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @instruction_alias_load() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %observed = load volatile i32, ptr inttoptr (i32 4293136384 to ptr), align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- remote-instruction-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @remote_instruction_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr inttoptr (i32 4293263360 to ptr), align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- donecheck-overlap.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @donecheck_overlap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i64 0, ptr inttoptr (i32 4293394440 to ptr), align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- l1-overlap.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @l1_overlap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr inttoptr (i32 1572862 to ptr), align 2
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- wallclock-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @wallclock_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr inttoptr (i32 4289798640 to ptr), align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- scalar-call.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @unclassified() memory(none)

define void @scalar_call() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @unclassified() memory(none)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

;--- inline-asm.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)

define void @inline_asm() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void asm sideeffect "", ""() memory(none)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}
