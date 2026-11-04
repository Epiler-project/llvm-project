; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/enabled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/predicated-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/recorded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/opaque.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/cfg-intersection.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/partial-rows.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/raw-constant.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/raw-constant.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/raw-loadi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/raw-loadi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/enabled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/predicated-old.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/recorded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-INGRESS
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/opaque.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/cfg-intersection.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/partial-rows.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID

; LaneConfig source bits come from executed contents, including inactive lanes.
; A zero ENABLE_DEST_INDEX bit preserves disabled indexing under any ROW_MASK
; and CC. It must not be inferred from a predicated immediate alone, a recorded
; instruction, or an SSA value invalidated by an opaque instruction-port write.
; INVALID: contents SFPU requires ENABLE_DEST_INDEX known disabled
; WH-INGRESS: Tensix intrinsic is not supported on Wormhole

;--- valid.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @disabled_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65535, i32 0)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65535, i32 1)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_mode2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_mode4() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65531, i32 4)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_mode8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 65535, i32 8)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_mode10() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

; Both the old contents and the newly written value have bit 2 clear, even
; though ROW_MASK=5 preserves rows 0 and 2 during the second LOADI.
define void @disabled_partial_rows() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %mask, i32 21845, i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %mask, i32 0, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @disabled_after_merge(i1 %choose) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  br i1 %choose, label %left, label %right
left:
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  br label %merge
right:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  br label %merge
merge:
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- enabled.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @enabled(ptr %port) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20484, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- unknown.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @unknown(ptr %port) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %old, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- predicated-old.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @predicated_old(ptr %port) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- recorded.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)

define void @recorded(ptr %port) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.replay.template.begin(i32 1, i32 0, i32 0, i32 0)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  call void @llvm.riscv.tt.replay.template.end(i32 1)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- opaque.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @opaque(ptr %port) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  store volatile i32 0, ptr %port, align 4
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- cfg-intersection.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)

; One predecessor loses contents knowledge. Its later LaneConfig reset must
; restore the hardware fact without restoring the earlier contents fact.
define void @cfg_intersection(i1 %choose, ptr %port) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  br i1 %choose, label %clean, label %opaque
clean:
  br label %merge
opaque:
  store volatile i32 0, ptr %port, align 4
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %merge
merge:
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  ret void
}

;--- partial-rows.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)

; ROW_MASK=5 preserves rows 0/2 of the known bit-2-one old value. In particular
; the first eight lanes still contain 4 when SFPCONFIG broadcasts them.
define void @partial_rows() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %enabled = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 4, i32 2)
  %mask.old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %mask.old, i32 20480, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %mask, i32 21845, i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %enabled, i32 20480, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  ret void
}

;--- raw-constant.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()

; A recorded MOVAll does not take a C9 snapshot until actually executed.
define void @raw_constant() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  ret void
}

;--- raw-loadi.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()

; Even with an executed zero seed, record-only LOADI defines no actual value.
define void @raw_loadi() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 20480, i32 2)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  ret void
}
