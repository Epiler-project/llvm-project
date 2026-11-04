; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -enable-new-pm -O0 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -enable-new-pm -O2 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; NEWPM: requires the SelectionDAG legacy codegen pipeline

; Contents remain raw e32 register bits. Loads and stores are observable SPM
; accesses; strict arithmetic preserves dynamic rounding and FP exceptions even
; when its vector result is unused. No physical selector enters the intrinsic.
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fadd(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsub(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmul(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fdiv(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmax(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmacc(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fcvt.to.i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fcvt.from.i32(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.eq(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.max(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare float @llvm.riscv.bosc.rvv.contents.fextract(<vscale x 2 x i32>, i64)
declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)

; CHECK-LABEL: strict_arithmetic:
; CHECK: vle32.v
; CHECK: vle32.v
; CHECK: vfadd.vv
; CHECK: vfsub.vv
; CHECK: vfmul.vv
; CHECK: vfdiv.vv
; CHECK: vfmax.vv
; CHECK: vfmacc.vv
; CHECK: vfcvt.x.f.v
; CHECK: vfcvt.f.x.v
; CHECK: vmflt.vv
; CHECK: vmfeq.vv
; CHECK: vse32.v
; MIR-LABEL: name: strict_arithmetic
; MIR: %{{[0-9]+}}:vr = PseudoVLE32_V_M1 {{.*}} :: (volatile load unknown-size
; MIR: PseudoVFADD_VV_M1_E32 {{.*}}implicit $frm
; MIR: PseudoVFDIV_VV_M1_E32 {{.*}}implicit $frm
; MIR: PseudoVSE32_V_M1 {{.*}} :: (volatile store unknown-size
define void @strict_arithmetic(ptr %a, ptr %b, ptr %out, i64 %n) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %add = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fadd(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %sub = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsub(<vscale x 2 x i32> poison, <vscale x 2 x i32> %add, <vscale x 2 x i32> %bv, i64 %vl)
  %mul = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmul(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sub, <vscale x 2 x i32> %bv, i64 %vl)
  %div = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fdiv(<vscale x 2 x i32> poison, <vscale x 2 x i32> %mul, <vscale x 2 x i32> %bv, i64 %vl)
  %max = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmax(<vscale x 2 x i32> poison, <vscale x 2 x i32> %div, <vscale x 2 x i32> %bv, i64 %vl)
  %macc = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fmacc(<vscale x 2 x i32> %max, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %asint = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fcvt.to.i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %macc, i64 %vl)
  %asfloat = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fcvt.from.i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %asint, i64 %vl)
  %lt = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %asfloat, <vscale x 2 x i32> %bv, i64 %vl)
  %eq = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.eq(<vscale x 2 x i1> poison, <vscale x 2 x i32> %asfloat, <vscale x 2 x i32> %bv, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %asfloat, ptr %out, i64 %vl)
  ret void
}

; CHECK-LABEL: strict_reduction:
; CHECK: vfmv.v.f
; CHECK: vle32.v
; CHECK: vfredosum.vs
; CHECK: vfredmax.vs
; CHECK: vfmv.f.s
; MIR-LABEL: name: strict_reduction
; MIR: PseudoVFREDOSUM_VS_M1_E32 {{.*}}implicit $frm
; MIR: PseudoVFREDMAX_VS_M1_E32
define float @strict_reduction(ptr %input, i64 %n, float %seed) "trex-bosc-contents-rvv"="1" {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float %seed, i64 %one)
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %vl)
  %max = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.max(<vscale x 2 x i32> %sum, <vscale x 2 x i32> %data, i64 %vl)
  %result = call float @llvm.riscv.bosc.rvv.contents.fextract(<vscale x 2 x i32> %max, i64 %vl)
  ret float %result
}

; CHECK-LABEL: observe_bits:
; CHECK: vsetivli zero, 1, e8, m1, ta, ma
; CHECK-NEXT: vle8.v
; CHECK-NEXT: vmv.x.s
; CHECK-NEXT: zext.b
; CHECK-NEXT: vsetvli zero, {{.*}}, e32, m1, ta, ma
; MIR-LABEL: name: observe_bits
; MIR: %{{[0-9]+}}:gprnox0, {{(dead )?}}%{{[0-9]+}}:vr = PseudoBOSCReadBitsBound {{.*}} :: (volatile load (s8)
define i64 @observe_bits(ptr %input, i64 %n) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %bits = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %input, i64 %vl, i32 8)
  ret i64 %bits
}
