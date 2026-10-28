; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %s | FileCheck %s --check-prefix=BH

declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.cleardvalid(i32 immarg, i32 immarg)

; This tests native logical forms and their issue order only. Reached ownership,
; cross-issuer matching, payload lifetime, and retaining-access completion are
; separate compiler obligations; the instructions alone do not establish them.
define void @clear_then_publish() "tensix-executor"="trisc0" {
; WH-LABEL: clear_then_publish:
; WH: .word 0x0c000045
; WH-NEXT: .word 0x0c00001d
; WH: .word 0x0e000045
; WH-NEXT: .word 0x0e00001d
; WH-NOT: call
; WH: ret
; ISEL-LABEL: name: clear_then_publish
; ISEL: TTUNPACR_NOP 1, 0, 0, 1, 0, 0, 0, 0, 0
; ISEL-NEXT: {{.*}}TTUNPACR_NOP 0, 0, 0, 0, 0, 1, 0, 0, 0
; BH-LABEL: clear_then_publish:
; BH: .word 0x0c000085
; BH-NEXT: .word 0x0c000401
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr.nop(i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.unpacr.nop(i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1)
  ret void
}

define void @return_source_lanes() "tensix-executor"="trisc1" {
; WH-LABEL: return_source_lanes:
; WH: .word 0xdc000004
; WH-NEXT: .word 0xd9000000
; WH: .word 0xdc000008
; WH-NEXT: .word 0xda000000
; WH-NOT: call
; WH: ret
; ISEL-LABEL: name: return_source_lanes
; ISEL: TTSETRWC 1, 0, 0, 0, 0, 0
; ISEL-NEXT: {{.*}}TTCLEARDVALID 0, 1
; ISEL: TTSETRWC 2, 0, 0, 0, 0, 0
; ISEL-NEXT: {{.*}}TTCLEARDVALID 0, 2
; BH-LABEL: return_source_lanes:
; BH: .word 0xdc000004
; BH-NEXT: .word 0xd9000000
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.cleardvalid(i32 0, i32 1)
  call void @llvm.riscv.tt.setrwc(i32 2, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.cleardvalid(i32 0, i32 2)
  ret void
}
