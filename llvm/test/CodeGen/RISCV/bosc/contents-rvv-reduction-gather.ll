; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -regalloc=fast -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -filetype=obj -verify-machineinstrs %s -o %t.o

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fadd(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.max(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)

; The reduction reads the seed and the complete actual active prefix. Only its
; result lane zero is needed by the later full-width broadcast, including when
; the dynamic reduction VL is zero. No physical selector or hidden copy exists.
; CHECK-LABEL: sum_broadcast:
; CHECK: vfmv.v.f
; CHECK: vle32.v
; CHECK: vfadd.vv
; CHECK: vfredosum.vs
; CHECK: vrgather.vi {{.*}}, 0
; CHECK: vse32.v
; MIR-LABEL: name: sum_broadcast
; MIR: PseudoVFREDOSUM_VS_M1_E32 {{.*}}implicit $frm
; MIR: early-clobber %{{[0-9]+}}:vr = PseudoVRGATHER_VI_M1
define void @sum_broadcast(ptr %input, ptr %output, float %seed, i64 %n) #0 {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float %seed, i64 %one)
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %twice = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fadd(<vscale x 2 x i32> poison, <vscale x 2 x i32> %data, <vscale x 2 x i32> %data, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %twice, i64 %vl)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %broadcast, ptr %output, i64 %full)
  ret void
}

; A VLMAX-initialized source also covers a different actual reduction VL.
; CHECK-LABEL: max_broadcast:
; CHECK: vle32.v
; CHECK: vfredmax.vs
; CHECK: vrgather.vi {{.*}}, 0
; CHECK: vse32.v
; MIR-LABEL: name: max_broadcast
; MIR: PseudoVFREDMAX_VS_M1_E32
; MIR: PseudoVRGATHER_VI_M1
define void @max_broadcast(ptr %input, ptr %output, float %seed, i64 %n) #0 {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float %seed, i64 %one)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %max = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.max(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %vl)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %max, i64 0, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %broadcast, ptr %output, i64 %full)
  ret void
}

; Loop-carried scalar contents remain in the actual reduction result register.
; The seed closes the PHI cycle without cloning a source iteration.
; CHECK-LABEL: sum_loop_broadcast:
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK: vfredosum.vs
; CHECK: b{{[a-z]+}} {{.*}}[[LOOP]]
; CHECK: vrgather.vi {{.*}}, 0
; CHECK: vse32.v
; MIR-LABEL: name: sum_loop_broadcast
; MIR: :vr = PHI
; MIR: PseudoVFREDOSUM_VS_M1_E32
; MIR: PseudoVRGATHER_VI_M1
define void @sum_loop_broadcast(ptr %input, ptr %output, float %seed, i64 %n, i64 %count) #0 {
entry:
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float %seed, i64 %one)
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi <vscale x 2 x i32> [ %initial, %entry ], [ %sum, %loop ]
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %acc, <vscale x 2 x i32> %data, i64 %vl)
  %inc = add i64 %i, 1
  %again = icmp ult i64 %inc, %count
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %broadcast, ptr %output, i64 %full)
  ret void
}

; VL=0 reads no data and leaves the initialized destination unchanged. Poison
; input must therefore be harmless without creating a new lane-zero value.
; CHECK-LABEL: zero_reduction:
; CHECK: vfmv.v.f
; CHECK: vrgather.vi {{.*}}, 0
; CHECK: vse32.v
; MIR-LABEL: name: zero_reduction
; MIR: PseudoVRGATHER_VI_M1
define void @zero_reduction(ptr %output, float %seed) #0 {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float %seed, i64 %one)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> poison, i64 0)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %broadcast, ptr %output, i64 %full)
  ret void
}

attributes #0 = { "trex-bosc-contents-rvv"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
