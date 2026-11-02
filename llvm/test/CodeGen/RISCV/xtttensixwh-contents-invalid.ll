; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bad-mad.ll -o %t/bad-mad.o 2>&1 | FileCheck %s --check-prefix=BAD_MAD
; RUN: test ! -e %t/bad-mad.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bad-mul.ll -o %t/bad-mul.o 2>&1 | FileCheck %s --check-prefix=BAD_MUL
; RUN: test ! -e %t/bad-mul.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bad-add.ll -o %t/bad-add.o 2>&1 | FileCheck %s --check-prefix=BAD_ADD
; RUN: test ! -e %t/bad-add.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bad-load.ll -o %t/bad-load.o 2>&1 | FileCheck %s --check-prefix=BAD_LOAD
; RUN: test ! -e %t/bad-load.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bad-store.ll -o %t/bad-store.o 2>&1 | FileCheck %s --check-prefix=BAD_STORE
; RUN: test ! -e %t/bad-store.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/dynamic-offset.ll -o %t/dynamic-offset.o 2>&1 | FileCheck %s --check-prefix=DYNAMIC_OFFSET
; RUN: test ! -e %t/dynamic-offset.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/bh-only.ll -o %t/bh-only.o 2>&1 | FileCheck %s --check-prefix=BH_ONLY
; RUN: test ! -e %t/bh-only.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/no-config.ll -o %t/no-config.o 2>&1 | FileCheck %s --check-prefix=NO_CONFIG
; RUN: test ! -e %t/no-config.o

; BAD_MAD: Wormhole SFPMAD operand 5 must be in [0, 0]
;--- bad-mad.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @bad_mad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %old, <32 x i32> %old, <32 x i32> %old, <32 x i32> %old, i32 1)
  ret void
}

; BAD_MUL: Wormhole SFPMUL operand 4 must be in [0, 0]
;--- bad-mul.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @bad_mul() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %old, <32 x i32> %old, <32 x i32> %old, i32 2)
  ret void
}

; BAD_ADD: Wormhole SFPADD operand 4 must be in [0, 0]
;--- bad-add.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @bad_add() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %old, <32 x i32> %old, <32 x i32> %old, i32 3)
  ret void
}

; BAD_LOAD: Wormhole SFPLOAD operand 3 must be in [0, 3]
;--- bad-load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_load() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 4, i32 3)
  ret void
}

; BAD_STORE: Wormhole SFPSTORE operand 2 must be in [0, 3]
;--- bad-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad_store() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 4, i32 3)
  ret void
}

; DYNAMIC_OFFSET: Wormhole contents Dst offset currently requires a static immediate
;--- dynamic-offset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @dynamic_offset(i32 %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  %result = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 %bounded, i32 0, i32 3)
  ret void
}

; BH_ONLY: Tensix instruction is not supported on Wormhole
;--- bh-only.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.arecip(<32 x i32>, <32 x i32>, i32 immarg)
define void @bh_only() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.arecip(<32 x i32> %old, <32 x i32> %old, i32 0)
  ret void
}

; NO_CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled
;--- no-config.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @no_config() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}
