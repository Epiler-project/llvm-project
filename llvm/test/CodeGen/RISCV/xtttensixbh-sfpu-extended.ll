; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=finalize-isel -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM

declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpsetexp.v(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfparecip(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd.i(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpshft2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
declare {<32 x i32>, <32 x i32>} @llvm.riscv.tt.sfpswap(<32 x i32>, <32 x i32>, i32 immarg)
declare {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfplut(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpstochrnd.i(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define void @independent_old() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: independent_old
; ISEL: TTSFPMOV
; ISEL: TTSFPSETEXP
; ISEL: TTSFPARECIP
; ISEL: TTSFPIADDI
; ISEL: TTSFPSHFT2
; ASM-LABEL: independent_old:
; ASM-NOT: call
; ASM: .word
; ASM: ret
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %base = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  %field = call <32 x i32> @llvm.riscv.tt.creg.read(i32 15)
  %set = call <32 x i32> @llvm.riscv.tt.sfpsetexp.v(
      <32 x i32> %old, <32 x i32> %base, <32 x i32> %field, i32 0)
  %recip = call <32 x i32> @llvm.riscv.tt.sfparecip(
      <32 x i32> %old, <32 x i32> %set, i32 0)
  %add = call <32 x i32> @llvm.riscv.tt.sfpiadd.i(
      <32 x i32> %recip, i32 -127, i32 5)
  %rotate = call <32 x i32> @llvm.riscv.tt.sfpshft2(
      <32 x i32> %old, <32 x i32> %add, i32 0, i32 3)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %field, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %rotate, i32 2, i32 0, i32 4)
  ret void
}

define void @multiple_results() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: multiple_results
; ISEL: TTSFPSWAP
; ISEL: TTSFPTRANSP
; ISEL-SAME: implicit-def $tt_l0, implicit-def $tt_l1, implicit-def $tt_l2, implicit-def $tt_l3
; ISEL-SAME: implicit-def dead $tt_l4, implicit-def dead $tt_l5, implicit-def dead $tt_l6, implicit-def dead $tt_l7
; ASM-LABEL: multiple_results:
; ASM: .word 0x30000002
; ASM: ret
  %a = call <32 x i32> @llvm.riscv.tt.creg.read(i32 8)
  %b = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %c = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  %d = call <32 x i32> @llvm.riscv.tt.creg.read(i32 15)
  %pair = call {<32 x i32>, <32 x i32>} @llvm.riscv.tt.sfpswap(<32 x i32> %a, <32 x i32> %b, i32 0)
  %p = extractvalue {<32 x i32>, <32 x i32>} %pair, 0
  %q = extractvalue {<32 x i32>, <32 x i32>} %pair, 1
  %four = call {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32> %p, <32 x i32> %q, <32 x i32> %c, <32 x i32> %d)
  %r0 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 0
  %r1 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 1
  %r2 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 2
  %r3 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 3
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r2, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r3, i32 3, i32 0, i32 4)
  ret void
}

define void @lut_and_round() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: lut_and_round
; ISEL: TTSFPLUT
; ISEL: TTSFPSTOCHRNDI
; ASM-LABEL: lut_and_round:
; ASM: .word 0x3c000002
; ASM: ret
  %a = call <32 x i32> @llvm.riscv.tt.creg.read(i32 8)
  %b = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %c = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  %d = call <32 x i32> @llvm.riscv.tt.creg.read(i32 15)
  %v = call <32 x i32> @llvm.riscv.tt.sfplut(<32 x i32> %b, <32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d, i32 4)
  %rounded = call <32 x i32> @llvm.riscv.tt.sfpstochrnd.i(<32 x i32> %b, <32 x i32> %v, i32 0, i32 6, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %rounded, i32 0, i32 0, i32 4)
  ret void
}
