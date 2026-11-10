; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/blackhole.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/blackhole.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wormhole.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wormhole.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -O0 %t/blackhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/wormhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/blackhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unbounded-blackhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/partial-blackhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/partial.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/outside-banks-blackhole.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/outside-banks.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/other-register.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/other-register.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/instruction-port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/opaque-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/opaque-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG

; A complete i32 write to the actual generation's bounded Ready/Reusable
; overlay registers preserves SFPU LaneConfig. These registers are overlay
; state, not aliases of a Tensix instruction port or SFPU configuration.
; Capacity bounds, exact register indices, access width, target features and
; opaque effects remain required. In particular, the opposite architecture's
; register pair must not become a store-preservation exception.
; CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled
; FEATURE: Tensix intrinsic requires +xtttensixbh

;--- blackhole.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @blackhole_stream_store(i32 range(i32 0, 64) %bank, i32 %counter) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 32
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 %counter, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  %ready.address = add i32 %base, 40
  %ready.pointer = inttoptr i32 %ready.address to ptr
  store volatile i32 %counter, ptr %ready.pointer, align 4
  %copied.again = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

; The producer's range is carried by a typed ABI load, through the actual
; RV32 cast chain. A loop must keep the entry reset across its store backedge.
define void @blackhole_stream_store_loop(ptr %slot, i32 %trips) "tensix-executor"="trisc1" {
entry:
  %loaded = load i32, ptr %slot, align 4, !range !0, !noundef !1
  %bank = freeze i32 %loaded
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 40
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %empty = icmp eq i32 %trips, 0
  br i1 %empty, label %exit, label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  fence seq_cst
  store volatile i32 %iteration, ptr %pointer, align 4
  %observed = load volatile i32, ptr %pointer, align 4
  fence seq_cst
  %next = add i32 %iteration, 1
  %more = icmp ult i32 %next, %trips
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- wormhole.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare void @llvm.riscv.tt.read.response(i32)

define void @wormhole_stream_store(i32 range(i32 0, 64) %bank, i32 %counter) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 12
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 %counter, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  %ready.address = add i32 %base, 16
  %ready.pointer = inttoptr i32 %ready.address to ptr
  store volatile i32 %counter, ptr %ready.pointer, align 4
  %copied.again = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

define void @wormhole_stream_store_loop(ptr %slot, i32 %trips) "tensix-executor"="trisc1" {
entry:
  %loaded = load i32, ptr %slot, align 4, !range !0, !noundef !1
  %bank = freeze i32 %loaded
  %bank64 = zext i32 %bank to i64
  %scaled = mul i64 %bank64, 4096
  %base = add i64 %scaled, 4289986560
  %address64 = add i64 %base, 16
  %address = trunc i64 %address64 to i32
  %wide = zext i32 %address to i64
  %pointer = inttoptr i64 %wide to ptr
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %empty = icmp eq i32 %trips, 0
  br i1 %empty, label %exit, label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  fence seq_cst
  store volatile i32 %iteration, ptr %pointer, align 4
  %observed = load volatile i32, ptr %pointer, align 4
  call void @llvm.riscv.tt.read.response(i32 %observed)
  fence seq_cst
  %next = add i32 %iteration, 1
  %more = icmp ult i32 %next, %trips
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- unknown.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @unknown_store(ptr %pointer) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- unbounded.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @unbounded_store(i32 %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 16
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- partial.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @partial_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %wh = inttoptr i32 4289986576 to ptr
  store volatile i16 0, ptr %wh, align 2
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- outside-banks.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @outside_store(i32 range(i32 0, 128) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 16
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- unbounded-blackhole.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @unbounded_store(i32 %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 40
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- partial-blackhole.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @partial_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %bh = inttoptr i32 4289986600 to ptr
  store volatile i16 0, ptr %bh, align 2
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- outside-banks-blackhole.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @outside_store(i32 range(i32 0, 128) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 40
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- other-register.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @other_register_store(i32 range(i32 0, 64) %bank) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %scaled = mul i32 %bank, 4096
  %base = add i32 %scaled, 4289986560
  %address = add i32 %base, 36
  %pointer = inttoptr i32 %address to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- instruction-port.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)

define void @instruction_port_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %pointer = inttoptr i32 4289728512 to ptr
  store volatile i32 0, ptr %pointer, align 4
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}

;--- opaque-call.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare void @opaque()

define void @opaque_call() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @opaque()
  %copied = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %value)
  ret void
}
