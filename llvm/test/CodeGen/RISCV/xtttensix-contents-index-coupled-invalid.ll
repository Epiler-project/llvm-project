; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/load-dead-index.ll -o %t/bh-O0-load-dead-index.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/bh-O0-load-dead-index.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/swap-dead-pair.ll -o %t/bh-O0-swap-dead-pair.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/bh-O0-swap-dead-pair.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/bh-O0-pressure.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/bh-O0-pressure.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/load-dead-index.ll -o %t/bh-O2-load-dead-index.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/bh-O2-load-dead-index.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/swap-dead-pair.ll -o %t/bh-O2-swap-dead-pair.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/bh-O2-swap-dead-pair.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/bh-O2-pressure.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/bh-O2-pressure.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/load-dead-index.ll -o %t/wh-O0-load-dead-index.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/wh-O0-load-dead-index.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/swap-dead-pair.ll -o %t/wh-O0-swap-dead-pair.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/wh-O0-swap-dead-pair.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/wh-O0-pressure.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/wh-O0-pressure.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/load-dead-index.ll -o %t/wh-O2-load-dead-index.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/wh-O2-load-dead-index.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/swap-dead-pair.ll -o %t/wh-O2-swap-dead-pair.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/wh-O2-swap-dead-pair.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/wh-O2-pressure.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/wh-O2-pressure.o

; A dead projected result does not permit overlapping mutable old roles.
; Tuples add no ninth leaf and no implicit numerical staging/move/spill.
; COPY: Tensix SFPU cannot synthesize a register copy
; PRESSURE: ran out of registers during register allocation

;--- load-dead-index.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

define void @load_shared_old_dead_index() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %old, <32 x i32> %old, i32 4, i32 0, i32 3)
  %r = extractvalue { <32 x i32>, <32 x i32> } %p, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 16, i32 0, i32 4)
  ret void
}

;--- swap-dead-pair.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

define void @swap_shared_old_dead_pair() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %idx = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %p = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %old, <32 x i32> %old, <32 x i32> %idx, <32 x i32> %idx, i32 1)
  %r = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %p, 0
  %i = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %p, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %i, i32 18, i32 0, i32 4)
  ret void
}

;--- pressure.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

define void @five_live_pairs() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %s0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p0 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s0, <32 x i32> %s1, i32 4, i32 0, i32 3)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %p0, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %p0, 1
  %s2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p1 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s2, <32 x i32> %s3, i32 8, i32 0, i32 3)
  %r2 = extractvalue { <32 x i32>, <32 x i32> } %p1, 0
  %r3 = extractvalue { <32 x i32>, <32 x i32> } %p1, 1
  %s4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p2 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s4, <32 x i32> %s5, i32 12, i32 0, i32 3)
  %r4 = extractvalue { <32 x i32>, <32 x i32> } %p2, 0
  %r5 = extractvalue { <32 x i32>, <32 x i32> } %p2, 1
  %s6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p3 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s6, <32 x i32> %s7, i32 16, i32 0, i32 3)
  %r6 = extractvalue { <32 x i32>, <32 x i32> } %p3, 0
  %r7 = extractvalue { <32 x i32>, <32 x i32> } %p3, 1
  %s8 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s9 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p4 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s8, <32 x i32> %s9, i32 20, i32 0, i32 3)
  %r8 = extractvalue { <32 x i32>, <32 x i32> } %p4, 0
  %r9 = extractvalue { <32 x i32>, <32 x i32> } %p4, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 42, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 44, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 46, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r8, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r9, i32 50, i32 0, i32 4)
  ret void
}
