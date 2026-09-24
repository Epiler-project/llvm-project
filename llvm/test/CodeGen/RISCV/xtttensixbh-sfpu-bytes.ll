; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.o0.o
; RUN: llvm-objdump -s -j .text %t.o0.o | FileCheck %s --check-prefix=BYTES
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.o2.o
; RUN: llvm-objdump -s -j .text %t.o2.o | FileCheck %s --check-prefix=BYTES
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: FileCheck %s --check-prefix=ASM < %t.o0.s
; RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t.o0.s -o %t.native-o0.o
; RUN: llvm-objdump -s -j .text %t.native-o0.o | FileCheck %s --check-prefix=BYTES
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: FileCheck %s --check-prefix=ASM < %t.o2.s
; RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t.o2.s -o %t.native-o2.o
; RUN: llvm-objdump -s -j .text %t.native-o2.o | FileCheck %s --check-prefix=BYTES

; Exact independently specified direct-issue words for enable, push, pop, NOP,
; followed by RV32 return. Textual assembly must use the actual MC encoding,
; and reassembling it must preserve these same bytes at both optimization levels.
; BYTES: 0000 2ac00028 0200001c 02000020 0200003c
; BYTES-NEXT: 0010 67800000
; ASM-LABEL: sfpu_control_bytes:
; ASM: .word 0x2800c02a
; ASM-NEXT: .word 0x1c000002
; ASM-NEXT: .word 0x20000002
; ASM-NEXT: .word 0x3c000002
; ASM-NEXT: ret

declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpnop()

define void @sfpu_control_bytes() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpnop()
  ret void
}
