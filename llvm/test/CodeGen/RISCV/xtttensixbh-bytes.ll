; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %s -o %t.direct.o
; RUN: llvm-objdump -s -j .text %t.direct.o | FileCheck %s --check-prefix=BYTES
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh %s -o %t.s
; RUN: FileCheck %s --check-prefix=ASM < %t.s
; RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t.s -o %t.native.o
; RUN: llvm-objdump -s -j .text %t.native.o | FileCheck %s --check-prefix=BYTES

; Both production paths have the same independently known instruction words.
; BYTES: 0000 0a0004c8 feff0fc9 67800000
; ASM: .word 0xc804000a
; ASM-NEXT: .word 0xc90ffffe

declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16(i32 2, i32 1)
  call void @llvm.riscv.tt.setc16(i32 65535, i32 67)
  ret void
}
