; PRNG is shared architectural state, including deterministic STOCHRND modes.
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpcast(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstochrnd.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @rounding_state() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: rounding_state
; CHECK: TTSFPCAST $tt_l0, $tt_c9, 1{{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; CHECK: TTSFPSTOCHRNDI $tt_l0, $tt_c9, 0, 0, 0{{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; CHECK: TTSFPSTOCHRNDI $tt_l0, $tt_c9, 0, 0, 2{{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; CHECK: TTSFPSTOCHRNDV $tt_l0, $tt_c9, $tt_c9, 4, 0{{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; CHECK: TTSFPSTOCHRNDV $tt_l0, $tt_c9, $tt_c9, 4, 2{{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpcast(i32 0, i32 0, i32 9, i32 1)
  call void @llvm.riscv.tt.bound.sfpstochrnd.i(i32 0, i32 0, i32 9, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstochrnd.i(i32 0, i32 0, i32 9, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpstochrnd.v(i32 0, i32 0, i32 9, i32 9, i32 4, i32 0)
  call void @llvm.riscv.tt.bound.sfpstochrnd.v(i32 0, i32 0, i32 9, i32 9, i32 4, i32 2)
  ret void
}
