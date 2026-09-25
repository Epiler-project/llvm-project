; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefixes=CHECK,O0
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefixes=CHECK,O2
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=PLAIN
;
; Physical reservation must not disable the existing legal optimizations.
; All data moves and numerical operations below are authored, including the
; snapshot in L2. Replay compresses repeated straight-line instructions; it
; does not duplicate any source iteration or synthesize an SFPU data action.

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpxor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpand(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpnop()

define void @bound_redundancy() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_redundancy
; CHECK: TTSFPENCC 3, 10,
; CHECK-NEXT: $tt_l0 = TTSFPMOVAll $tt_c9,
; CHECK-NEXT: $tt_l0 = TTSFPMOVNeg $tt_l0, $tt_l0,
; CHECK-NOT: TTSFPMOV
; CHECK: TTSFPSTORE $tt_l0,
; CHECK: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

define void @bound_keep_toggle_and_barrier() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: bound_keep_toggle_and_barrier
; CHECK: TTSFPENCC 0, 1,
; CHECK-NEXT: TTSFPENCC 0, 1,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpnop()
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  ret void
}

define void @bound_latency() "tensix-executor"="trisc2" {
; CHECK-LABEL: name: bound_latency
; CHECK: $tt_l0 = TTSFPMAD
; O0-NEXT: TTSFPNOP
; O0-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; O0-NEXT: $tt_l4 = TTSFPMOVAll $tt_c9,
; O2-NEXT: $tt_l4 = TTSFPMOVAll $tt_c9,
; O2-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK: TTSFPSTORE $tt_l0,
; CHECK: TTSFPSTORE $tt_l4,
; CHECK: PseudoRET
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

define void @bound_replay_snapshot() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_replay_snapshot
; CHECK: $tt_l2 = TTSFPMOVAll $tt_l0,
; CHECK: TTSFPPUSHC
; CHECK: TTSFPSETCCNE
; CHECK: PseudoTTSFPUReplay 1, 1, 6, 0,
; CHECK-NEXT: $tt_l0 = TTSFPIADD
; CHECK-NEXT: $tt_l0 = TTSFPXOR
; CHECK-NEXT: $tt_l0 = TTSFPAND
; CHECK-NEXT: $tt_l0 = TTSFPOR
; CHECK-NEXT: $tt_l0 = TTSFPXOR
; CHECK-NEXT: $tt_l0 = TTSFPIADD
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTSFPUReplay 0, 0, 6, 0,
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPPOPC
; CHECK: TTSFPSTORE $tt_l0, 2, 0, 4,
; CHECK: TTSFPSTORE $tt_l2, 3, 0, 4,
; CHECK: PseudoRET
; PLAIN-LABEL: name: bound_replay_snapshot
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSETCCNE
; PLAIN-NEXT: $tt_l0 = TTSFPIADD
; PLAIN-NEXT: $tt_l0 = TTSFPXOR
; PLAIN-NEXT: $tt_l0 = TTSFPAND
; PLAIN-NEXT: $tt_l0 = TTSFPOR
; PLAIN-NEXT: $tt_l0 = TTSFPXOR
; PLAIN-NEXT: $tt_l0 = TTSFPIADD
; PLAIN-NEXT: $tt_l0 = TTSFPIADD
; PLAIN-NEXT: $tt_l0 = TTSFPXOR
; PLAIN-NEXT: $tt_l0 = TTSFPAND
; PLAIN-NEXT: $tt_l0 = TTSFPOR
; PLAIN-NEXT: $tt_l0 = TTSFPXOR
; PLAIN-NEXT: $tt_l0 = TTSFPIADD
; PLAIN-NEXT: TTSFPPOPC
; PLAIN: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 1, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 1, i32 4)
  call void @llvm.riscv.tt.bound.sfpxor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpand(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpxor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 1, i32 4)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 1, i32 4)
  call void @llvm.riscv.tt.bound.sfpxor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpand(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpxor(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 1, i32 4)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 3, i32 0, i32 4)
  ret void
}
