; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/cell.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CELL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/brisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BRISC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON

; ISEL-LABEL: name: constant_controls
; ISEL: PseudoTTMOPControlWriteImm 0, 1, {{.*}} :: (volatile store (s32))
; ISEL: PseudoTTMOPControlWriteImm 1, 305419896, {{.*}} :: (volatile store (s32))
; ISEL-LABEL: name: dynamic_control
; ISEL: PseudoTTMOPControlWrite 0, {{.*}} :: (volatile store (s32))
; ASM-LABEL: constant_controls:
; ASM: lui [[ADDRESS:[a-z0-9]+]], 1047424
; ASM: sw {{[a-z0-9]+}}, 0([[ADDRESS]])
; ASM: lui [[VALUE:[a-z0-9]+]], 74565
; ASM: addi [[VALUE]], [[VALUE]], 1656
; ASM: lui [[ADDRESS2:[a-z0-9]+]], 1047424
; ASM: sw [[VALUE]], 4([[ADDRESS2]])
; ASM-LABEL: bit_patterns:
; ASM: sw zero, 0({{[a-z0-9]+}})
; ASM: lui [[ALL:[a-z0-9]+]], 0
; ASM: addi [[ALL]], [[ALL]], -1
; ASM: sw [[ALL]], 4({{[a-z0-9]+}})
; ASM: lui [[LOW:[a-z0-9]+]], 1
; ASM: addi [[LOW]], [[LOW]], -2048
; ASM: sw [[LOW]], 0({{[a-z0-9]+}})
; ASM-LABEL: dynamic_control:
; ASM: lui [[ADDRESS3:[a-z0-9]+]], 1047424
; ASM: sw a0, 0([[ADDRESS3]])
; CELL: must be in [0, 1]
; BRISC: MOP control programming requires a TRISC executor
; POISON: MOP control value must be defined and non-poison

;--- valid.ll
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
define void @constant_controls() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 305419896)
  ret void
}
define void @bit_patterns() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 0)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 -1)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 2048)
  ret void
}
define i32 @dynamic_control(i32 noundef %value) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 %value)
  ret i32 %value
}

;--- cell.ll
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
define void @bad_cell() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.mop.control.write(i32 2, i32 1)
  ret void
}

;--- brisc.ll
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
define void @remote() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  ret void
}

;--- poison.ll
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
define void @poisoned() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 poison)
  ret void
}
