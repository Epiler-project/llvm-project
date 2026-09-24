; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=finalize-isel -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM

declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)

define void @ordinary_static() #0 {
; ISEL-LABEL: name: ordinary_static
; ISEL: TTSTALLWAIT 7, 8
; ISEL: TTSETRWC 63, 1, 2, 3, 4, 0
; ISEL: TTSETC16 65535, 67
; ISEL: TTNOP
; ASM-LABEL: ordinary_static:
; ASM: .word 0x8810001e
; ASM: .word 0xdc4321fc
; ASM: .word 0xc90ffffe
; ASM: .word 0x08000000
  call void @llvm.riscv.tt.stallwait(i32 7, i32 8)
  call void @llvm.riscv.tt.setrwc(i32 63, i32 1, i32 2, i32 3, i32 4, i32 0)
  call void @llvm.riscv.tt.setc16(i32 65535, i32 67)
  call void @llvm.riscv.tt.nop()
  ret void
}

define void @ordinary_dynamic(i32 noundef %value) #0 {
; ISEL-LABEL: name: ordinary_dynamic
; ISEL: PseudoTTSTALLWAITPort
; ASM-LABEL: ordinary_dynamic:
; ASM: slli
; ASM: or
; ASM: sw
  %wait = and i32 %value, 32767
  %stall = and i32 %value, 511
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 %wait, i32 %stall)
  ret void
}

attributes #0 = { "tensix-executor"="trisc1" }
