; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/source.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SOURCE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh %t/odd-mask.ll -o - | FileCheck %s --check-prefix=MASK
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh %t/odd-mask-wh.ll -o - | FileCheck %s --check-prefix=MASK
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/negative.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/imm-value-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; SOURCE: bound SFPU fixed register group does not match the instruction
; MASK: .word 0x44000be2
; WIDTH: Tensix immediate operand must fit in 16 unsigned bits
; MODE: SFPCONFIGLane mode must be one of {8}

;--- source.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @wrong_source() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 1, i32 21844, i32 8)
  ret void
}
;--- odd-mask.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @odd_mask() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 0, i32 2, i32 8)
  ret void
}
;--- overflow.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @overflow() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 0, i32 65536, i32 8)
  ret void
}
;--- negative.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @negative() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 0, i32 -1, i32 8)
  ret void
}
;--- mode.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @unsupported_mode() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 0, i32 0, i32 0)
  ret void
}
;--- imm-value-mode.ll
declare void @llvm.tt.bh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @unsupported_immediate_data() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.config.lane(i32 0, i32 21844, i32 9)
  ret void
}

;--- odd-mask-wh.ll
declare void @llvm.tt.wh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)
define void @odd_mask_wh() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.config.lane(i32 0, i32 2, i32 8)
  ret void
}
