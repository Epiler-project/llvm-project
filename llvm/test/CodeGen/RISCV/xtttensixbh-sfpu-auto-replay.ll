; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=SELECT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=SELECT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=PLAIN
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=PLAIN
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

; This is authored straight-line arithmetic, not an unrolled source loop.
; Only the machine selector can create the typed automatic replay pseudo.
; The ordinary explicit-recording rejection remains in replay-invalid.ll.
; Both inputs are independent Dst values. The original full-lane value remains
; live after the masked arithmetic, so replay must preserve its allocation.
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32>, <32 x i32>)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @masked_arithmetic_with_snapshot() "tensix-executor"="trisc1" {
; SELECT-LABEL: name: masked_arithmetic_with_snapshot
; SELECT: TTSFPLOAD
; SELECT: TTSFPLOAD
; SELECT: TTSFPPUSHC
; SELECT: TTSFPSETCCNE
; SELECT: PseudoTTSFPUReplay 1, 1, 6, 0,
; SELECT-NEXT: {{.*}} = TTSFPIADD
; SELECT-NEXT: {{.*}} = TTSFPXOR
; SELECT-NEXT: {{.*}} = TTSFPAND
; SELECT-NEXT: {{.*}} = TTSFPOR
; SELECT-NEXT: {{.*}} = TTSFPXOR
; SELECT-NEXT: {{.*}} = TTSFPIADD
; SELECT-NEXT: PseudoTTReplayRecordEnd
; SELECT-NEXT: TTSFPNOP
; SELECT-NEXT: PseudoTTSFPUReplay 0, 0, 6, 0,
; SELECT-NEXT: TTSFPNOP
; SELECT-NEXT: TTSFPPOPC
; SELECT: TTSFPSTORE {{.*}}, 2, 0, 4,
; SELECT: TTSFPSTORE {{.*}}, 3, 0, 4,
; SELECT: PseudoRET
; PLAIN-LABEL: name: masked_arithmetic_with_snapshot
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSETCCNE
; PLAIN: TTSFPIADD
; PLAIN-NEXT: {{.*}} = TTSFPXOR
; PLAIN-NEXT: {{.*}} = TTSFPAND
; PLAIN-NEXT: {{.*}} = TTSFPOR
; PLAIN-NEXT: {{.*}} = TTSFPXOR
; PLAIN-NEXT: {{.*}} = TTSFPIADD
; PLAIN-NEXT: {{.*}} = TTSFPIADD
; PLAIN-NEXT: {{.*}} = TTSFPXOR
; PLAIN-NEXT: {{.*}} = TTSFPAND
; PLAIN-NEXT: {{.*}} = TTSFPOR
; PLAIN-NEXT: {{.*}} = TTSFPXOR
; PLAIN-NEXT: {{.*}} = TTSFPIADD
; PLAIN-NEXT: TTSFPPOPC
; PLAIN: TTSFPSTORE {{.*}}, 2, 0, 4,
; PLAIN: TTSFPSTORE {{.*}}, 3, 0, 4,
; PLAIN: PseudoRET
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 0, i32 0, i32 4)
  %rhs = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %rhs, i32 0, i32 2)
  %a0 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %old, <32 x i32> %rhs, i32 4)
  %a1 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %a0, <32 x i32> %rhs)
  %a2 = call <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32> %a1, <32 x i32> %rhs)
  %a3 = call <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32> %a2, <32 x i32> %rhs)
  %a4 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %a3, <32 x i32> %rhs)
  %a5 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %a4, <32 x i32> %rhs, i32 4)
  %b0 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %a5, <32 x i32> %rhs, i32 4)
  %b1 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %b0, <32 x i32> %rhs)
  %b2 = call <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32> %b1, <32 x i32> %rhs)
  %b3 = call <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32> %b2, <32 x i32> %rhs)
  %b4 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %b3, <32 x i32> %rhs)
  %b5 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %b4, <32 x i32> %rhs, i32 4)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %b5, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %old, i32 3, i32 0, i32 4)
  ret void
}
