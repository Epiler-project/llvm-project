; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/range.ll -o %t/range.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/range.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/addrmod.ll -o %t/addrmod.o 2>&1 | FileCheck %s --check-prefix=ADDRMOD
; RUN: test ! -e %t/addrmod.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/format.ll -o %t/format.o 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: test ! -e %t/format.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/poison.ll -o %t/poison.o 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: test ! -e %t/poison.o

; A dynamic coordinate only changes how the instruction is issued. It does
; not widen the ten-bit Dst field, the two-bit WH AddrMod, or legal formats,
; and cannot turn a potentially poison source into a defined coordinate.
; RANGE: contents Dst offset must be proven in [0, 1023]
; ADDRMOD: Wormhole SFPSTORE operand 2 must be in [0, 3]
; FORMAT: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
; POISON: contents Dst offset must be defined and non-poison

;--- range.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @wide_offset(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 2047
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed, i32 %bounded, i32 0, i32 3)
  ret void
}

;--- addrmod.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @blackhole_addrmod(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed, i32 %bounded, i32 4, i32 3)
  ret void
}

;--- format.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @reserved_format(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 %bounded, i32 0, i32 7)
  ret void
}

;--- poison.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @possibly_poison(i32 %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed, i32 %bounded, i32 0, i32 3)
  ret void
}
