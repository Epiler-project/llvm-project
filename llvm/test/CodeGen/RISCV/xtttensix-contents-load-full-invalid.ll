; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-bh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/no-reset.ll -o %t/no-reset-wh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/no-reset-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/reset-before-enable.ll -o %t/order-bh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/order-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/reset-before-enable.ll -o %t/order-wh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/order-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/masked-after-reset.ll -o %t/masked-bh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/masked-after-reset.ll -o %t/masked-wh-o0.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/masked-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/one-predecessor-reset.ll -o %t/join-bh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/join-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/one-predecessor-reset.ll -o %t/join-wh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/join-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/row-mask.ll -o %t/row-bh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/row-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/row-mask.ll -o %t/row-wh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/row-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/read-block.ll -o %t/block-bh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/block-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/read-block.ll -o %t/block-wh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/block-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/unknown-store.ll -o %t/store-bh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/store-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/unknown-store.ll -o %t/store-wh.o 2>&1 | FileCheck %s --check-prefix=STATE
; RUN: not test -e %t/store-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/low-half.ll -o %t/low-bh-o0.o 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not test -e %t/low-bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/low-half.ll -o %t/low-wh-o0.o 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not test -e %t/low-wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/high-half.ll -o %t/high-bh.o 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not test -e %t/high-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/high-half.ll -o %t/high-wh.o 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not test -e %t/high-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/partial-fake-old.ll -o %t/old-bh.o 2>&1 | FileCheck %s --check-prefix=OLD
; RUN: not test -e %t/old-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/partial-fake-old.ll -o %t/old-wh.o 2>&1 | FileCheck %s --check-prefix=OLD
; RUN: not test -e %t/old-wh.o

; Full LOAD is a checked full definition, not permission to forget old bits.
; Effective CC alone cannot prove ROW_MASK or Dst read-block configuration.
; These failures leave no object, and ordinary partial LOAD still rejects a
; fabricated numerical old operand.
; STATE: contents full LOAD requires all lanes writable
; FORMAT: contents full LOAD requires a complete-word format
; OLD: SFPU carrier constants require explicit target initialization

;--- no-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @encc_does_not_clear_row_or_read_masks() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- reset-before-enable.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @a_cold_masked_reset_is_not_full_setup() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- masked-after-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @later_lane_predication_requires_actual_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- one-predecessor-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @every_incoming_path_must_establish_full_setup(i1 %choose) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  br i1 %choose, label %left, label %join
left:
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
join:
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- row-mask.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @row_mask_is_independent_of_effective_cc() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %row = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %zero, i32 4096, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %row, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- read-block.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @unblocked_reads_are_an_independent_requirement() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %block = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %zero, i32 32, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %block, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- unknown-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @unknown_instruction_port_alias_invalidates_setup(ptr %unknown) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr %unknown
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- low-half.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @opaque_low_half_requires_actual_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 14)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- high-half.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @opaque_high_half_requires_actual_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- partial-fake-old.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @partial_load_cannot_invent_an_old_value() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> poison, i32 0, i32 0, i32 14)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}
