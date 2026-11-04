; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/mode.ll -o %t/bh-mode.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: test ! -e %t/bh-mode.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/bh-undef.o 2>&1 | FileCheck %s --check-prefix=UNDEF
; RUN: test ! -e %t/bh-undef.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/alias.ll -o %t/bh-alias.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/bh-alias.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/mode.ll -o %t/wh-mode.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: test ! -e %t/wh-mode.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/undef.ll -o %t/wh-undef.o 2>&1 | FileCheck %s --check-prefix=UNDEF
; RUN: test ! -e %t/wh-undef.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/alias.ll -o %t/wh-alias.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/wh-alias.o

; The index form adds its complete tuple; it does not add repair capacity.
; Duplicating a single writable SSA value into two different singleton roles
; must diagnose rather than invent a numerical staging copy.
; MODE: contents swap index mode must be in 0..9
; UNDEF: SFPU carrier constants require explicit target initialization
; COPY: Tensix SFPU cannot synthesize a register copy

;--- mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)


define void @invalid_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 10)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}

;--- undef.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)


define void @invalid_undef() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> undef, <32 x i32> %a3, i32 9)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}

;--- alias.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)


define void @invalid_alias() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a1, <32 x i32> %a3, i32 9)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}
