; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.fsplat(i32 immarg, float)
declare void @llvm.riscv.bosc.freduce.sum(i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.freduce.max(i32 immarg, i32 immarg)
declare float @llvm.riscv.bosc.fextract(i32 immarg)

; CHECK-LABEL: sum:
; CHECK: vsetivli {{.*}}, 1, e32, m1, ta, ma
; CHECK: vfmv.v.f v4
; CHECK: vsetvli
; CHECK: vle32.v v5
; CHECK: vfredosum.vs v4, v5, v4
; CHECK: vfmv.f.s {{.*}}, v4
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: sum
; MIR: $v4 = VFMV_V_F
; MIR: $v4 = VFREDOSUM_VS $v5, $v4, $noreg, implicit $frm, implicit $vl, implicit $vtype
; MIR: VFMV_F_S $v4, implicit $vtype
define float @sum(ptr %input, i64 %n, float %seed) "trex-bosc-bound-rvv"="1" {
  %initvl = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.fsplat(i32 4, float %seed)
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %n)
  call void @llvm.riscv.bosc.load(i32 5, ptr %input)
  call void @llvm.riscv.bosc.freduce.sum(i32 4, i32 5)
  %result = call float @llvm.riscv.bosc.fextract(i32 4)
  ret float %result
}

; CHECK-LABEL: max:
; CHECK: vfredmax.vs v4, v5, v4
; MIR-LABEL: name: max
; MIR: $v4 = VFREDMAX_VS $v5, $v4, $noreg, implicit $vl, implicit $vtype
define float @max(ptr %input, float %seed) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.fsplat(i32 4, float %seed)
  call void @llvm.riscv.bosc.load(i32 5, ptr %input)
  call void @llvm.riscv.bosc.freduce.max(i32 4, i32 5)
  %result = call float @llvm.riscv.bosc.fextract(i32 4)
  ret float %result
}
