; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %s | FileCheck %s --check-prefix=BH

declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @source_bank_handoff() "tensix-executor"="trisc0" {
; WH-LABEL: source_bank_handoff:
; WH: .word 0x88100402
; WH-NEXT: .word 0x0c00001d
; WH: .word 0x88100802
; WH-NEXT: .word 0x0e00001d
; WH-NOT: call
; WH: ret
; ISEL-LABEL: name: source_bank_handoff
; ISEL: TTSTALLWAIT 256, 8
; ISEL-NEXT: {{.*}}TTUNPACR_NOP 0, 0, 0, 0, 0, 1, 0, 0, 0
; ISEL: TTSTALLWAIT 512, 8
; ISEL-NEXT: {{.*}}TTUNPACR_NOP 0, 0, 0, 0, 0, 1, 0, 0, 1
; BH-LABEL: source_bank_handoff:
; BH: .word 0x0c000401
; BH: .word 0x0e000401
  ; STALLWAIT logical operand order is condition bits, block bits.
  ; C8/C9 wait for the selected unpacker bank to belong to Unpackers.
  ; C10/C11 instead wait for Matrix-owned banks and cannot acquire writable
  ; ownership. The bare handoff itself performs no wait and writes no payload.
  call void @llvm.riscv.tt.stallwait(i32 256, i32 8)
  call void @llvm.riscv.tt.unpacr.nop(i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.stallwait(i32 512, i32 8)
  call void @llvm.riscv.tt.unpacr.nop(i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1)
  ret void
}
