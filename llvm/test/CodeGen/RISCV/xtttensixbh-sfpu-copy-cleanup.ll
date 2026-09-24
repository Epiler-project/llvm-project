; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=ELIMINATED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=ELIMINATED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

; These are selected target cleanups, not a claim that generic RA coalesces
; every copy. ENCC modes 0/2/8/10 are idempotent; modes 1/9 toggle enable.
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpnop()

define void @duplicate_encc() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: duplicate_encc
; ELIMINATED: TTSFPENCC 0, 0,
; ELIMINATED-NEXT: TTSFPENCC 1, 2,
; ELIMINATED-NEXT: TTSFPENCC 2, 8,
; ELIMINATED-NEXT: TTSFPENCC 3, 10,
; ELIMINATED-NOT: TTSFPENCC
; ELIMINATED: PseudoRET
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}

define void @self_copy_mode0() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: self_copy_mode0
; ELIMINATED: TTSFPMOVAll $tt_c9
; ELIMINATED-NOT: TTSFPMOV
; ELIMINATED: TTSFPSTORE
; ELIMINATED-NOT: TTSFPMOV
; ELIMINATED: PseudoRET
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %copy = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %copy, i32 0, i32 0, i32 4)
  ret void
}

define void @self_copy_all() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: self_copy_all
; ELIMINATED: TTSFPMOVAll $tt_c9
; ELIMINATED-NOT: TTSFPMOV
; ELIMINATED: TTSFPSTORE
; ELIMINATED-NOT: TTSFPMOV
; ELIMINATED: PseudoRET
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %copy = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 2)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %copy, i32 0, i32 0, i32 4)
  ret void
}

define void @keep_negation() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: keep_negation
; ELIMINATED: TTSFPMOVNeg
; ELIMINATED: TTSFPSTORE
; ELIMINATED: PseudoRET
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %neg = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 1)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %neg, i32 0, i32 0, i32 4)
  ret void
}

define void @keep_toggle_and_barrier() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: keep_toggle_and_barrier
; ELIMINATED: TTSFPENCC 0, 1,
; ELIMINATED-NEXT: TTSFPENCC 0, 1,
; ELIMINATED-NEXT: TTSFPENCC 2, 9,
; ELIMINATED-NEXT: TTSFPENCC 2, 9,
; ELIMINATED-NEXT: TTSFPENCC 3, 10,
; ELIMINATED-NEXT: TTSFPNOP
; ELIMINATED-NEXT: TTSFPENCC 3, 10,
; ELIMINATED: PseudoRET
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}

; Fixed reads are stable snapshots. The write between reads and the old-value
; use after a masked assignment must remain distinct in the final stream.
define void @keep_snapshots_and_masked_later_use() "tensix-executor"="trisc1" {
; ELIMINATED-LABEL: name: keep_snapshots_and_masked_later_use
; ELIMINATED: TTSFPMOVAll $tt_l3
; ELIMINATED: $tt_l3 = TTSFPMOVAll
; ELIMINATED: TTSFPMOVAll $tt_l3
; ELIMINATED: TTSFPSETCCNE
; ELIMINATED: TTSFPMOV {{.*}}, implicit $tt_cc
; ELIMINATED: TTSFPSTORE
; ELIMINATED: TTSFPSTORE
; ELIMINATED: TTSFPENCC 3, 10,
; ELIMINATED: PseudoRET
  %before = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %zero)
  %after = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %after, i32 0, i32 2)
  %selected = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %before, <32 x i32> %after, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %selected, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %before, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}
