; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM

declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)

; Modes three and four peek at the saved frame without consuming it. Both
; original frames still require a plain pop, and only CC/issue are modified.
define void @combine_modes() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: combine_modes
; ISEL: TTSFPPUSHC
; ISEL-NEXT: TTSFPPUSHC
; ISEL-NEXT: TTSFPPOPCAndTop implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue{{$}}
; ISEL-NEXT: TTSFPPOPCOrTop implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue{{$}}
; ISEL-NEXT: TTSFPPOPC implicit
; ISEL-NEXT: TTSFPPOPC implicit
; ASM-LABEL: combine_modes:
; ASM: .word 0x1c000002
; ASM-NEXT: .word 0x1c000002
; ASM-NEXT: .word 0x2000000e
; ASM-NEXT: .word 0x20000012
; ASM-NEXT: .word 0x20000002
; ASM-NEXT: .word 0x20000002
; ASM-NEXT: ret
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  ret void
}
