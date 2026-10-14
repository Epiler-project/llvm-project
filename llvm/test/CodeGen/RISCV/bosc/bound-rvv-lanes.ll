; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s

declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.add(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; Whole 31-element Tasks use independently bound 16- and 15-lane roots.
; Configuration changes do not clobber another root's initialized prefix.
; CHECK-LABEL: whole_31:
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vle32.v v5
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v6
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vle32.v v7
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vadd.vv v4, v4, v6
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vadd.vv v5, v5, v7
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vse32.v v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vse32.v v5
; CHECK-NOT: vmv
; CHECK-NOT: vs1r
; CHECK-NOT: vl1r
define void @whole_31(ptr %a, ptr %b, ptr %out) #0 {
  %at = getelementptr i32, ptr %a, i64 16
  %bt = getelementptr i32, ptr %b, i64 16
  %ot = getelementptr i32, ptr %out, i64 16
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 5, ptr %at)
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 6, ptr %b)
  %vl3 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 7, ptr %bt)
  %vl4 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.add(i32 4, i32 4, i32 6)
  %vl5 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.add(i32 5, i32 5, i32 7)
  %vl6 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %out)
  %vl7 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.store(i32 5, ptr %ot)
  ret void
}

; VL=0 does not execute elements or perform tail-agnostic writes.
; CHECK-LABEL: zero_write:
; CHECK: vle32.v v4
; CHECK: vsetivli {{.*}}, 0, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vse32.v v4
define void @zero_write(ptr %p) #0 {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 0)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}

; A 15-lane read is valid on either incoming path; the 16th lane is not.
; CHECK-LABEL: join_prefix:
; CHECK: vle32.v v4
; CHECK: vle32.v v4
; CHECK: vse32.v v4
define void @join_prefix(ptr %p, i1 %cond) #0 {
entry:
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br i1 %cond, label %short, label %join
short:
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br label %join
join:
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}

; The preheader deliberately follows the loop in block order. Reaching
; configurations start at bottom while initialized prefixes start at top.
; CHECK-LABEL: loop_prefix:
; CHECK-DAG: vle32.v v4
; CHECK-DAG: vadd.vv v4, v4, v4
; CHECK-DAG: vse32.v v4
; CHECK-DAG: {{bnez|bne|j}} {{.*}}.LBB
define void @loop_prefix(ptr %p, i1 %again) #0 {
entry:
  br label %preheader
loop:
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.add(i32 4, i32 4, i32 4)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  br i1 %again, label %loop, label %exit, !llvm.loop !0
preheader:
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br label %loop
exit:
  ret void
}

attributes #0 = { "trex-bosc-bound-rvv"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
