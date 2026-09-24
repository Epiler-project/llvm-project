; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/signed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SIGNED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/setfield.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SETFIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/shiftmode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHIFTMODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/shuffleimm.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHUFFLEIMM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/floatdescale.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FLOATDESCALE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vectorfloat.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORFLOAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/swapmode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SWAPMODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/dstformat.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DSTFORMAT

; DSTFORMAT: unsupported instruction mode 16 for 'llvm.riscv.tt.sfpload'
;--- dstformat.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 0, i32 7, i32 16)
  ret void
}

; SIGNED: signed immediate operand 1 must be in [-2048, 2047]
;--- signed.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd.i(<32 x i32>, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpiadd.i(<32 x i32> %v, i32 2048, i32 5)
  ret void
}

; SETFIELD: immediate operand 2 must be in [0, 255]
;--- setfield.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpsetexp.i(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpsetexp.i(<32 x i32> %v, <32 x i32> %v, i32 256, i32 1)
  ret void
}

; SHIFTMODE: unsupported instruction mode 4
;--- shiftmode.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpshft.v(<32 x i32>, <32 x i32>, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpshft.v(<32 x i32> %v, <32 x i32> %v, i32 4)
  ret void
}

; SHUFFLEIMM: immediate operand 2 must be in [0, 0]
;--- shuffleimm.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpshft2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpshft2(<32 x i32> %v, <32 x i32> %v, i32 1, i32 3)
  ret void
}

; FLOATDESCALE: immediate operand 2 must be in [0, 0]
;--- floatdescale.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpstochrnd.i(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpstochrnd.i(<32 x i32> %v, <32 x i32> %v, i32 1, i32 6, i32 0)
  ret void
}

; VECTORFLOAT: unsupported instruction mode 6
;--- vectorfloat.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpstochrnd.v(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.sfpstochrnd.v(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 6, i32 0)
  ret void
}

; SWAPMODE: immediate operand 2 must be in [0, 9]
;--- swapmode.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare {<32 x i32>, <32 x i32>} @llvm.riscv.tt.sfpswap(<32 x i32>, <32 x i32>, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %r = call {<32 x i32>, <32 x i32>} @llvm.riscv.tt.sfpswap(<32 x i32> %v, <32 x i32> %v, i32 10)
  ret void
}
