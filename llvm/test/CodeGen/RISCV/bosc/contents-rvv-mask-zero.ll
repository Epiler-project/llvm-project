; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=vmv1r --implicit-check-not=vl1r --implicit-check-not=vs1r
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=vmv1r --implicit-check-not=vl1r --implicit-check-not=vs1r
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.eq(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)

; With zero VL, compare writes no destination bits. The initialized mask must
; survive in the same physical register and remain consumable after VL grows.
; CHECK-LABEL: zero_mask:
; CHECK: vmfeq.vv v0,
; CHECK: vmflt.vv v0,
; CHECK: vmerge.vvm {{.*}}, v0
; CHECK: vse32.v
; MIR-LABEL: name: zero_mask
; MIR: %[[OLD:[0-9]+]]:vr = PseudoVMFEQ_VV_M1
; MIR: PseudoBOSC_VMFLT_VV_M1 %[[OLD]],
define void @zero_mask(ptr %a, ptr %b, ptr %out) "trex-bosc-contents-rvv"="1" {
  %full = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %full)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %full)
  %old = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.eq(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %full)
  %zero = call i64 @llvm.riscv.vsetvli.i64(i64 0, i64 2, i64 0)
  %next = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> %old, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %zero)
  %restored = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %selected = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %next, i64 %restored)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %selected, ptr %out, i64 %restored)
  ret void
}

; A runtime zero receives the same tie. A positive runtime VL makes mask tails
; agnostic, so the widened observation is guarded by the actual zero result.
; CHECK-LABEL: dynamic_zero_mask:
; CHECK: vmfeq.vv v0,
; CHECK: vmflt.vv v0,
; CHECK: vmerge.vvm {{.*}}, v0
; CHECK: vse32.v
; MIR-LABEL: name: dynamic_zero_mask
; MIR: %[[DOLD:[0-9]+]]:vr = PseudoVMFEQ_VV_M1
; MIR: PseudoBOSC_VMFLT_VV_M1 %[[DOLD]],
define void @dynamic_zero_mask(ptr %a, ptr %b, ptr %out, i64 %n) "trex-bosc-contents-rvv"="1" {
entry:
  %full = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %full)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %full)
  %old = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.eq(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %full)
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %next = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> %old, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %zero = icmp eq i64 %vl, 0
  br i1 %zero, label %observe, label %done
observe:
  %restored = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %selected = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %next, i64 %restored)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %selected, ptr %out, i64 %restored)
  br label %done
done:
  ret void
}
