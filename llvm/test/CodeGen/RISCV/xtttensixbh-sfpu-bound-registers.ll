; Bound operations enter ISel with physical numbers, never an SFPR SSA result.
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefixes=BOUND-PHYSICAL,NO-VIRTUAL-SFPR,EXPLICIT-OLD-VALUE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefixes=BOUND-PHYSICAL,NO-VIRTUAL-SFPR,EXPLICIT-OLD-VALUE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NO-SPILL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NO-SPILL

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

; Final words follow the ISA opcode/field encoding (rotate-left two bits),
; independently of the generated opcode table. NEXT checks exclude repair ops.
; The only copy is explicitly authored here. The multiply must not force a
; repair move, and its result has the exact physical destination at ISel exit.
define void @bound_load_mul_store() "tensix-executor"="trisc1" {
; BOUND-PHYSICAL-LABEL: name: bound_load_mul_store
; NO-VIRTUAL-SFPR-NOT: class: sfpr
; NO-VIRTUAL-SFPR-NOT: PseudoTTBound
; BOUND-PHYSICAL: $tt_l0 = TTSFPLOAD $tt_l0, 0, 0, 3
; BOUND-PHYSICAL: $tt_l1 = TTSFPLOAD $tt_l1, 4, 0, 3
; BOUND-PHYSICAL: $tt_l2 = TTSFPMOVAll $tt_c9
; EXPLICIT-OLD-VALUE: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 0
; BOUND-PHYSICAL: TTSFPSTORE $tt_l2, 8, 0, 3
; NO-VIRTUAL-SFPR-NOT: class: sfpr
; NO-SPILL-LABEL: bound_load_mul_store:
; NO-SPILL-NOT: sw
; NO-SPILL: .word 0x2800c02a
; NO-SPILL-NEXT: .word 0xc00c0001
; NO-SPILL-NEXT: .word 0xc04c0011
; NO-SPILL-NEXT: .word 0xf0002489
; NO-SPILL-NEXT: .word 0x18006482
; NO-SPILL-NEXT: .word 0xc88c0021
; NO-SPILL-NEXT: ret
; NO-SPILL-NOT: lw
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 3)
  ret void
}
