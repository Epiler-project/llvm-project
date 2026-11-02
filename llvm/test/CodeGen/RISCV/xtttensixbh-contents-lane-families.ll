; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; Ordinary single-result SFPU families receive numerical contents, including old
; destination passthrough, implicit CC/config/PRNG effects, and vector fields
; whose value is necessarily the old destination. No authored register number
; appears in a numerical operation. Each fixture consumes its result in Dst.

declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.prng.advance(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.arecip(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.abs(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.pushc(i32 immarg)
declare void @llvm.riscv.tt.contents.popc(i32 immarg)
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.gt(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.le(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.compc()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.contents.config.reset()

define void @contents_load_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_load_immediate>:
; CHECK: ttsfploadi
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_load_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %zero, i32 7, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_mad() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_mad>:
; CHECK: ttsfpmad
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_mad
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %zero, <32 x i32> %one, <32 x i32> %one, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_iadd() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_iadd>:
; CHECK: ttsfpiadd
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_iadd
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %zero, <32 x i32> %one, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_mov() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_mov>:
; CHECK: ttsfpmov.neg
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_mov
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32> %zero, <32 x i32> %one, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_mov_all() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_mov_all>:
; CHECK: ttsfpmov.all
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_mov_all
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mov.all(<32 x i32> %one)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_prng_advance() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_prng_advance>:
; CHECK: ttsfpmov.prng.advance
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_prng_advance
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.prng.advance(<32 x i32> %zero)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_arecip() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_arecip>:
; CHECK: ttsfparecip
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_arecip
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.arecip(<32 x i32> %zero, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_exexp() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_exexp>:
; CHECK: ttsfpexexp
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_exexp
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %zero, <32 x i32> %one, i32 8)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_exman() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_exman>:
; CHECK: ttsfpexman
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_exman
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %zero, <32 x i32> %one, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_abs() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_abs>:
; CHECK: ttsfpabs
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_abs
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.abs(<32 x i32> %zero, <32 x i32> %one, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_lz() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_lz>:
; CHECK: ttsfplz
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_lz
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %zero, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_cast() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_cast>:
; CHECK: ttsfpcast
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_cast
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %zero, <32 x i32> %one, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setexp_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setexp_immediate>:
; CHECK: ttsfpsetexp
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setexp_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %zero, <32 x i32> %one, i32 127, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setman_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setman_immediate>:
; CHECK: ttsfpsetman
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setman_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %zero, <32 x i32> %one, i32 123, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setsgn_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setsgn_immediate>:
; CHECK: ttsfpsetsgn
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setsgn_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %zero, <32 x i32> %one, i32 1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setexp_vector() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setexp_vector>:
; CHECK: ttsfpsetexp
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setexp_vector
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %zero, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setman_vector() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setman_vector>:
; CHECK: ttsfpsetman
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setman_vector
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32> %zero, <32 x i32> %one, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_setsgn_vector() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_setsgn_vector>:
; CHECK: ttsfpsetsgn
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_setsgn_vector
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %zero, <32 x i32> %one, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_iadd_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_iadd_immediate>:
; CHECK: ttsfpiadd.i
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_iadd_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %zero, i32 -1, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_shft_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_shft_immediate>:
; CHECK: ttsfpshft
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_shft_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %zero, i32 3, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_shft_vector() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_shft_vector>:
; CHECK: ttsfpshft
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_shft_vector
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %zero, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_and() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_and>:
; CHECK: ttsfpand
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_and
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %zero, <32 x i32> %one)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_or() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_or>:
; CHECK: ttsfpor
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_or
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %zero, <32 x i32> %one)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_xor() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_xor>:
; CHECK: ttsfpxor
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_xor
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %zero, <32 x i32> %one)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_not() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_not>:
; CHECK: ttsfpnot
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_not
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %zero)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_shft2() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_shft2>:
; CHECK: ttsfpshft2
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_shft2
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32> %zero, <32 x i32> %one, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_stochrnd_immediate() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_stochrnd_immediate>:
; CHECK: ttsfpstochrnd.i
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_stochrnd_immediate
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32> %zero, <32 x i32> %one, i32 7, i32 4, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_stochrnd_vector() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_stochrnd_vector>:
; CHECK: ttsfpstochrnd.v
; CHECK: ttsfpstore
; ISEL-LABEL: name: contents_stochrnd_vector
; ISEL: :sfpr = TTSFPMOVAll
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32> %zero, <32 x i32> %one, <32 x i32> %one, i32 5, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 7, i32 3)
  ret void
}

define void @contents_predicate_state() "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_predicate_state>:
; CHECK: ttsfpencc
; CHECK: ttsfppushc
; CHECK: ttsfppushc.replace_top
; CHECK: ttsfpsetcc.eq0
; CHECK: ttsfpgt
; CHECK: ttsfple
; CHECK: ttsfpcompc
; CHECK: ttsfppopc.peek
; CHECK: ttsfppopc.and_top
; CHECK: ttsfppopc.or_top
; CHECK: ttsfppopc
; CHECK: ttsfpnop
; CHECK: ttsfpconfig.reset
; ISEL-LABEL: name: contents_predicate_state
; ISEL: TTSFPGT {{.*}}implicit-def{{( dead)?}} $tt_cc
; ISEL: TTSFPLE {{.*}}implicit-def{{( dead)?}} $tt_cc
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.pushc(i32 0)
  call void @llvm.riscv.tt.contents.pushc(i32 1)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %zero, i32 6)
  call void @llvm.riscv.tt.contents.gt(<32 x i32> %zero, <32 x i32> %one, i32 1)
  call void @llvm.riscv.tt.contents.le(<32 x i32> %zero, <32 x i32> %one, i32 1)
  call void @llvm.riscv.tt.contents.compc()
  call void @llvm.riscv.tt.contents.popc(i32 1)
  call void @llvm.riscv.tt.contents.popc(i32 3)
  call void @llvm.riscv.tt.contents.popc(i32 4)
  call void @llvm.riscv.tt.contents.popc(i32 0)
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.config.reset()
  ret void
}
