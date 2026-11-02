; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+zve32x,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -filetype=obj -verify-machineinstrs %s -o %t.o

; Raw e32 integer contents have actual use/def and the same explicit VL and
; issue ordering as ordered FP/SPM operations. No register selector is passed.
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.add(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.maximum(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.isplat(<vscale x 2 x i32>, i64, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.iadd.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.shl(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)

; CHECK-LABEL: integer_contents:
; CHECK: vle32.v
; CHECK: vmv.v.x
; CHECK: vadd.vv
; CHECK: vmax.vv
; CHECK: vadd.vx
; CHECK: vsll.vi {{.*}}, 31
; CHECK: vse32.v
; MIR-LABEL: name: integer_contents
; MIR: PseudoVMV_V_X_M1
; MIR: PseudoVADD_VV_M1
; MIR: PseudoVMAX_VV_M1
; MIR: PseudoVADD_VX_M1
; MIR: PseudoVSLL_VI_M1
define void @integer_contents(ptr %input, ptr %output, i64 %n, i64 %scalar) #0 {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.isplat(<vscale x 2 x i32> poison, i64 %scalar, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.add(<vscale x 2 x i32> poison, <vscale x 2 x i32> %data, <vscale x 2 x i32> %broadcast, i64 %vl)
  %max = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.maximum(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, <vscale x 2 x i32> %data, i64 %vl)
  %adjusted = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.iadd.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %max, i64 %scalar, i64 %vl)
  %shifted = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.shl(<vscale x 2 x i32> poison, <vscale x 2 x i32> %adjusted, i64 31, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %shifted, ptr %output, i64 %vl)
  ret void
}

; Lane 15 is defined by the VLMAX load and remains readable as a gather source
; with VL=1. Gather source indices are checked against VLMAX, not active VL.
; CHECK-LABEL: gather_inactive_source_lane:
; CHECK: vle32.v [[SOURCE:v[0-9]+]],
; CHECK: vrgather.vi [[DEST:v[0-9]+]], [[SOURCE]], 15
; CHECK: vse32.v [[DEST]],
; MIR-LABEL: name: gather_inactive_source_lane
; MIR: early-clobber %{{[0-9]+}}:vr = PseudoVRGATHER_VI_M1
define void @gather_inactive_source_lane(ptr %input, ptr %output) #0 {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %gather = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %data, i64 15, i64 %one)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %gather, ptr %output, i64 %one)
  ret void
}

; CHECK-LABEL: integer_loop:
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK: vadd.vx
; CHECK: b{{[a-z]+}} {{.*}}[[LOOP]]
; CHECK: vse32.v
; MIR-LABEL: name: integer_loop
; MIR: PHI
; MIR: PseudoVADD_VX_M1
define void @integer_loop(ptr %input, ptr %output, i64 %n, i64 %count, i64 %step) #0 {
entry:
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %contents = phi <vscale x 2 x i32> [ %initial, %entry ], [ %updated, %loop ]
  %updated = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.iadd.scalar(<vscale x 2 x i32> %contents, <vscale x 2 x i32> %contents, i64 %step, i64 %vl)
  %next = add i64 %i, 1
  %again = icmp ult i64 %next, %count
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %updated, ptr %output, i64 %vl)
  ret void
}

; The gather's own positive-VL path guarantees source lane zero when both
; operations receive the same actual VL; VL=0 reads no source lane.
; CHECK-LABEL: gather_dynamic_zero:
; CHECK: vle32.v
; CHECK: vrgather.vi {{.*}}, 0
; CHECK: vse32.v
; MIR-LABEL: name: gather_dynamic_zero
; MIR: PseudoVRGATHER_VI_M1
define void @gather_dynamic_zero(ptr %input, ptr %output, i64 %n) #0 {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %gather = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %data, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %gather, ptr %output, i64 %vl)
  ret void
}

; Even tail-agnostic operations preserve all old contents when VL=0.
; CHECK-LABEL: gather_after_zero_vl:
; CHECK: vle32.v
; CHECK: vrgather.vi {{.*}}, 15
; CHECK: vse32.v
; MIR-LABEL: name: gather_after_zero_vl
; MIR: PseudoVRGATHER_VI_M1
define void @gather_after_zero_vl(ptr %input, ptr %output) #0 {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  %preserved = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> %data, ptr %input, i64 0)
  %gather = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %preserved, i64 15, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %gather, ptr %output, i64 %full)
  ret void
}

; A seeded PHI cycle retains initialized lanes without cloning the loop.
; CHECK-LABEL: gather_loop_contents:
; CHECK: [[GATHER_LOOP:.LBB[0-9_]+]]:
; CHECK: vadd.vx
; CHECK: b{{[a-z]+}} {{.*}}[[GATHER_LOOP]]
; CHECK: vrgather.vi {{.*}}, 15
; CHECK: vse32.v
; MIR-LABEL: name: gather_loop_contents
; MIR: PHI
; MIR: PseudoVRGATHER_VI_M1
define void @gather_loop_contents(ptr %input, ptr %output, i64 %count) #0 {
entry:
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %contents = phi <vscale x 2 x i32> [ %initial, %entry ], [ %updated, %loop ]
  %updated = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.iadd.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %contents, i64 1, i64 %full)
  %next = add i64 %i, 1
  %again = icmp ult i64 %next, %count
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %gather = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %updated, i64 15, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %gather, ptr %output, i64 %full)
  ret void
}

attributes #0 = { "trex-bosc-contents-rvv"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
