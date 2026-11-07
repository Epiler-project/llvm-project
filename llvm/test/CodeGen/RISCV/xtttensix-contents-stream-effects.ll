; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/fence.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/fence.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/fence.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/fence.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/stream.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/stream.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/stream.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/stream.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/combined.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/combined.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/combined.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/combined.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/wrong-register.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/wrong-register.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/partial.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/partial.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/cross-bank.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/cross-bank.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/config-write.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/config-write.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG

; Only ordering and the exact typed stream-counter address domain preserve
; SFPU LaneConfig. No default configuration or arbitrary MMIO exception.
; CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled

;--- fence.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @fence_only() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  fence seq_cst
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- stream.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @stream_counter_12(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 12
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

define void @stream_counter_16(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 16
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

define void @stream_counter_32(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 32
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

define void @stream_counter_40(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 40
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}


;--- combined.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @typed_abi_stream(ptr %slot) "tensix-executor"="trisc1" {
  %loaded = load i32, ptr %slot, align 4, !range !0, !noundef !1
  %bank = freeze i32 %loaded
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  fence seq_cst
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 32
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  fence seq_cst
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}


!0 = !{i32 0, i32 64}
!1 = !{}
;--- unknown.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @unknown_pointer(ptr %pointer) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  fence seq_cst
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- unbounded.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @unbounded_bank(i32 %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 32
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- wrong-register.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @different_register(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 36
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- partial.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @partial_counter(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 32
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i16, ptr %pointer, align 2
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- cross-bank.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @outside_inventory(i32 range(i32 0, 128) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 32
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- config-write.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.read.response(i32)
declare void @llvm.assume(i1 noundef)

define void @real_config_write() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %value, i32 4, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  fence seq_cst
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}
