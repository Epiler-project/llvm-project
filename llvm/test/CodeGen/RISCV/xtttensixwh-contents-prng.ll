; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/positive.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/positive.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/positive.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/round-immediate.ll -o %t/round-immediate.o 2>&1 | FileCheck %s --check-prefix=ROUND-I
; RUN: test ! -e %t/round-immediate.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %t/round-vector.ll -o %t/round-vector.o 2>&1 | FileCheck %s --check-prefix=ROUND-V
; RUN: test ! -e %t/round-vector.o
;
; Shared native descriptors admit WH PRNG advance and STOCHRND through the
; numerical contents receiver too. Values remain SSA and reach Dst through the
; existing contents path. All rounding modes retain PRNG read/write effects.
;
; ROUND-I: Wormhole SFPSTOCHRNDI operand 5 must be in [0, 1]
; ROUND-V: Wormhole SFPSTOCHRNDV operand 5 must be in [0, 1]

;--- positive.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.prng.advance(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg, i32 immarg)

; Each active lane receives the updated PRNG state, preserving inactive old.
define void @contents_prng_advance() "tensix-executor"="trisc1" {
; NATIVE-LABEL: <contents_prng_advance>:
; NATIVE: ttsfpmov.prng.advance
; NATIVE: ttsfpstore
; ISEL-LABEL: name: contents_prng_advance
; ISEL: :sfpr = TTSFPMOVPRNGAdvance
; ISEL-SAME: implicit-def{{( dead)?}} $tt_prng
; ISEL-SAME: implicit $tt_prng
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.prng.advance(<32 x i32> %zero)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 3, i32 3)
  ret void
}

define void @contents_stochrnd_immediate() "tensix-executor"="trisc1" {
; NATIVE-LABEL: <contents_stochrnd_immediate>:
; NATIVE: ttsfpstochrnd.i
; NATIVE: ttsfpstore
; ISEL-LABEL: name: contents_stochrnd_immediate
; ISEL: :sfpr = TTSFPSTOCHRNDI {{.*}}, 7, 12, 1,
; ISEL-SAME: implicit-def{{( dead)?}} $tt_prng
; ISEL-SAME: implicit $tt_prng
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32> %zero, <32 x i32> %one, i32 7, i32 4, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 3, i32 3)
  ret void
}

define void @contents_stochrnd_vector_nearest() "tensix-executor"="trisc1" {
; NATIVE-LABEL: <contents_stochrnd_vector_nearest>:
; NATIVE: ttsfpstochrnd.v
; NATIVE: ttsfpstore
; ISEL-LABEL: name: contents_stochrnd_vector_nearest
; ISEL: :sfpr = TTSFPSTOCHRNDV {{.*}}, 5, 0,
; ISEL-SAME: implicit-def{{( dead)?}} $tt_prng
; ISEL-SAME: implicit $tt_prng
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32> %zero, <32 x i32> %one, <32 x i32> %one, i32 5, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 3, i32 3)
  ret void
}

;--- round-immediate.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @round_immediate() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.immediate(<32 x i32> %zero, <32 x i32> %zero, i32 0, i32 0, i32 2)
  ret void
}

;--- round-vector.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @round_vector() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.stochrnd.vector(<32 x i32> %zero, <32 x i32> %zero, <32 x i32> %zero, i32 4, i32 2)
  ret void
}
