; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -regalloc=fast -filetype=obj -verify-machineinstrs %s -o %t.fast.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -riscv-rvv-regalloc=fast -filetype=obj -verify-machineinstrs %s -o %t.rvv-fast.o

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)

; The pseudo configures e32,m1 itself. A valid earlier actual VL remains valid
; after an unrelated configuration; this is not the fixed bound-state ABI.
; CHECK-LABEL: reuse_actual_vl:
; CHECK: vle32.v
; CHECK: vse32.v
define void @reuse_actual_vl(ptr %p, ptr %out, i64 %n) "trex-bosc-contents-rvv"="1" {
  %actual = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %other = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 0, i64 1)
  store volatile i64 %other, ptr %out
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %actual)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %v, ptr %out, i64 %actual)
  ret void
}

; CHECK-LABEL: phi_actual_vl:
; CHECK: vle32.v
; CHECK: vse32.v
define void @phi_actual_vl(ptr %p, ptr %out, i64 %n, i1 %choose) "trex-bosc-contents-rvv"="1" {
entry:
  br i1 %choose, label %bounded, label %maximum
bounded:
  %actual = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  br label %join
maximum:
  %max = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  br label %join
join:
  %vl = phi i64 [ %actual, %bounded ], [ %max, %maximum ]
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %v, ptr %out, i64 %vl)
  ret void
}

; A real loop carries its actual VL through a PHI. It retains the source loop
; and remains legal when the next actual value differs on each iteration.
; CHECK-LABEL: loop_actual_vl:
; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: vle32.v
; CHECK: vse32.v
; CHECK: b{{[a-z]+}} {{.*}}.LBB{{[0-9]+}}_1
define void @loop_actual_vl(ptr %p, ptr %out, i64 %n, i64 %count) "trex-bosc-contents-rvv"="1" {
entry:
  %initial = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  br label %loop
loop:
  %vl = phi i64 [ %initial, %entry ], [ %next, %loop ]
  %remaining = phi i64 [ %count, %entry ], [ %decremented, %loop ]
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %v, ptr %out, i64 %vl)
  %decremented = sub i64 %remaining, 1
  %next = call i64 @llvm.riscv.vsetvli.i64(i64 %decremented, i64 2, i64 0)
  %more = icmp ne i64 %decremented, 0
  br i1 %more, label %loop, label %done
done:
  ret void
}

; The observation has a private one-element configuration and restores VL0.
; Its width is independent of the public e32,m1 restore value.
; CHECK-LABEL: observe_without_configuration:
; CHECK: vsetivli zero, 1, e64, m1, ta, ma
; CHECK: vle64.v
; CHECK: vmv.x.s
; CHECK: vsetvli zero, {{.*}}, e32, m1, ta, ma
define i64 @observe_without_configuration(ptr %p) "trex-bosc-contents-rvv"="1" {
  %bits = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %p, i64 0, i32 64)
  ret i64 %bits
}
