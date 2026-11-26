; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/wh.ll -o - | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs %t/bh.ll -o - | FileCheck %s --check-prefix=BH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/bh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh,+xtttensixbh %t/wh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE-CONFLICT
; RUN: not llc -mtriple=riscv32 %t/wh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/round.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROUND
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/mad.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MAD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/unsupported.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN-ABI
; MISMATCH: SFPU intrinsic architecture does not match target feature
; FEATURE-CONFLICT: XTTTensixBH and XTTTensixWH are mutually exclusive
; FEATURE: Tensix intrinsic requires +xtttensixbh or +xtttensixwh
; ROUND: Wormhole SFPSTOCHRNDI operand 5 must be in [0, 1]
; MAD: Wormhole SFPMAD operand 5 must be in [0, 0]
; UNKNOWN-ABI: unsupported Tensix intrinsic ABI: llvm.tt.wh.sfpu.arecip

; The native word check distinguishes WH's zero unused VB field from BH's VC
; scheduling operand. Both architectures keep round 0=nearest, 1=stochastic.
;--- wh.ll
declare void @llvm.tt.wh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov.all(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov.prng.advance(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.stochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @wormhole() "tensix-executor"="trisc1" {
; WH-LABEL: wormhole:
; WH: .word 0xf0002421
; WH: .word 0x38000402
; WH: .word 0x38800402
; WH-NOT: llvm.tt
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 10)
  call void @llvm.tt.wh.sfpu.mov.prng.advance(i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.stochrnd.i(i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.stochrnd.i(i32 0, i32 0, i32 1, i32 0, i32 0, i32 1)
  ret void
}
;--- bh.ll
declare void @llvm.tt.bh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.mov.all(i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.stochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @blackhole() "tensix-executor"="trisc2" {
; BH-LABEL: blackhole:
; BH: .word 0x38004402
; BH: .word 0x39004402
; BH-NOT: llvm.tt
  call void @llvm.tt.bh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.bh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.bh.sfpu.mov.all(i32 1, i32 10)
  call void @llvm.tt.bh.sfpu.stochrnd.i(i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  call void @llvm.tt.bh.sfpu.stochrnd.i(i32 0, i32 0, i32 1, i32 0, i32 0, i32 2)
  ret void
}
;--- round.ll
declare void @llvm.tt.wh.sfpu.stochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @round() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.stochrnd.i(i32 0, i32 0, i32 9, i32 0, i32 0, i32 2)
  ret void
}
;--- mad.ll
declare void @llvm.tt.wh.sfpu.mad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @mad() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.mad(i32 0, i32 0, i32 9, i32 10, i32 9, i32 1)
  ret void
}
;--- unsupported.ll
declare void @llvm.tt.wh.sfpu.arecip(i32, i32, i32, i32)
define void @unsupported() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.arecip(i32 0, i32 0, i32 9, i32 0)
  ret void
}
