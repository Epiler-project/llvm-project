; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE --implicit-check-not=fence --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE --implicit-check-not=fence --implicit-check-not=call
;
; Blackhole SETC16(CFG_STATE_ID) must not share a fused frontend bundle with
; a following instruction that depends on the new bank. At most four adjacent
; direct Tensix instructions fuse, so three authored, bank-independent TTNOPs
; separate the selector from RMWCIB0 even when the selector starts a bundle.
; TTNOP accesses no Auto-TTSync resources; it is not an SFPU-local SFPNOP.
;
; This codegen regression catches removal/reordering of the authored TTNOPs or
; insertion of an unrequested fence/wait/repair instruction. The physical L0
; value crosses the ordinary configuration sequence without a temporary or a
; spill. The state selector and its required padding are source-authored.
;
; This is not TT program admission or a hardware completion test. In particular,
; the compile-only controls below deliberately omit enough padding for a reliable
; fusion boundary. Native emission preserves that authored sequence without
; inserting padding or proving its hardware scheduling correctness.

declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.rmwcib0(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

; Native direct issue rotates the raw ISA word left by two bits:
; SETC16(value=0,reg=0): rol32(0xb2000000,2) = 0xc8000002.
; TTNOP: rol32(0x02000000,2) = 0x08000000.
; RMWCIB0(cfgregaddr=6,data=0,mask=255):
; rol32(0xb3000000 | (255 << 16) | 6,2) = 0xcffc001a.
; This writes only Config6's low byte; it is not complete Dst initialization.
; SFPMOV(all,L0,C9): rol32(0x7c000902,2) = 0xf0002409.
; SFPMOV(all,L1,L0): rol32(0x7c000012,2) = 0xf0000049.
define void @authored_config_boundary() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: authored_config_boundary
; ISEL: registers: []
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: TTSETC16 0, 0,
; ISEL-NEXT: TTNOP
; ISEL-NEXT: TTNOP
; ISEL-NEXT: TTNOP
; ISEL-NEXT: TTRMWCIB0 6, 0, 255,
; ISEL-NEXT: $tt_l1 = TTSFPMOVAll $tt_l0,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: authored_config_boundary:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002409
; NATIVE-NEXT: .word 0xc8000002
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0xcffc001a
; NATIVE-NEXT: .word 0xf0000049
; NATIVE-NEXT: ret
; NATIVE-NOT: .word
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.setc16(i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.rmwcib0(i32 6, i32 0, i32 255)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 0)
  ret void
}

; No padding is authored. Immediate RMWCIB0 emission is a negative witness:
; LLVM has neither established a safe fusion boundary nor repaired the source.
define void @omitted_config_boundary() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: omitted_config_boundary
; ISEL: registers: []
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: TTSETC16 0, 0,
; ISEL-NEXT: TTRMWCIB0 6, 0, 255,
; ISEL-NEXT: $tt_l1 = TTSFPMOVAll $tt_l0,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: omitted_config_boundary:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002409
; NATIVE-NEXT: .word 0xc8000002
; NATIVE-NEXT: .word 0xcffc001a
; NATIVE-NEXT: .word 0xf0000049
; NATIVE-NEXT: ret
; NATIVE-NOT: .word
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.setc16(i32 0, i32 0)
  call void @llvm.riscv.tt.rmwcib0(i32 6, i32 0, i32 255)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 0)
  ret void
}

; Two NOPs still allow selector and consumer in the same four-word bundle.
; Preserve the incomplete authored sequence rather than adding the third NOP.
define void @insufficient_config_boundary() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: insufficient_config_boundary
; ISEL: registers: []
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: TTSETC16 0, 0,
; ISEL-NEXT: TTNOP
; ISEL-NEXT: TTNOP
; ISEL-NEXT: TTRMWCIB0 6, 0, 255,
; ISEL-NEXT: $tt_l1 = TTSFPMOVAll $tt_l0,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: insufficient_config_boundary:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002409
; NATIVE-NEXT: .word 0xc8000002
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0xcffc001a
; NATIVE-NEXT: .word 0xf0000049
; NATIVE-NEXT: ret
; NATIVE-NOT: .word
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.setc16(i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.rmwcib0(i32 6, i32 0, i32 255)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 0)
  ret void
}
