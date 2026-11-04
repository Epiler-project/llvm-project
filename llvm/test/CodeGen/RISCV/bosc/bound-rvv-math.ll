; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.fmax(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fmacc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fcmp.lt(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.merge(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fcvt.to.i32(i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fcvt.from.i32(i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.iadd.scalar(i32 immarg, i32 immarg, i64)
declare void @llvm.riscv.bosc.shl(i32 immarg, i32 immarg, i32 immarg)
; CHECK-LABEL: math:
; CHECK: vfmax.vv v6, v4, v5
; CHECK: vfmacc.vv v6, v4, v5
; CHECK: vmflt.vv v0, v4, v5
; CHECK: vmerge.vvm v6, v4, v5, v0
; CHECK: vfcvt.x.f.v v6, v6
; CHECK: vadd.vx v6, v6
; CHECK: vsll.vi v6, v6, 23
; CHECK: vfcvt.f.x.v v5, v4
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: math
; MIR: $v6 = VFMACC_VV $v6, $v4, $v5, $noreg
; MIR: $v0 = VMFLT_VV $v4, $v5, $noreg
; MIR: $v6 = VMERGE_VVM $v4, $v5, $v0
define void @math(ptr %a, ptr %b, ptr %out) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.fmax(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.fmacc(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.fcmp.lt(i32 0, i32 4, i32 5)
  call void @llvm.riscv.bosc.merge(i32 6, i32 4, i32 5, i32 0)
  call void @llvm.riscv.bosc.fcvt.to.i32(i32 6, i32 6)
  call void @llvm.riscv.bosc.iadd.scalar(i32 6, i32 6, i64 127)
  call void @llvm.riscv.bosc.shl(i32 6, i32 6, i32 23)
  call void @llvm.riscv.bosc.fcvt.from.i32(i32 5, i32 4)
  call void @llvm.riscv.bosc.store(i32 6, ptr %out)
  ret void
}
