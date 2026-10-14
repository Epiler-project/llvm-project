; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; Scalar pointers and actual VL remain LLVM SSA/GPR values. Each source loop
; iteration stays a loop iteration and uses the actual VL for all advances.
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.add(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
; CHECK-LABEL: stripmine:
; CHECK: .LBB0_{{[0-9]+}}:
; CHECK: vsetvli [[VL:[a-z][0-9]+]], {{[a-z][0-9]+}}, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vle32.v v5
; CHECK: vadd.vv v6, v4, v5
; CHECK: vse32.v v6
; CHECK: {{sub|add}}
; CHECK: {{bnez|bne|j}} {{.*}}.LBB0_
; CHECK-NOT: vmv
define void @stripmine(ptr %a, ptr %b, ptr %c, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %loop
loop:
  %left = phi i64 [ %n, %entry ], [ %next, %loop ]
  %ap = phi ptr [ %a, %entry ], [ %an, %loop ]
  %bp = phi ptr [ %b, %entry ], [ %bn, %loop ]
  %cp = phi ptr [ %c, %entry ], [ %cn, %loop ]
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %left)
  call void @llvm.riscv.bosc.load(i32 4, ptr %ap)
  call void @llvm.riscv.bosc.load(i32 5, ptr %bp)
  call void @llvm.riscv.bosc.add(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 6, ptr %cp)
  %an = getelementptr i32, ptr %ap, i64 %vl
  %bn = getelementptr i32, ptr %bp, i64 %vl
  %cn = getelementptr i32, ptr %cp, i64 %vl
  %next = sub i64 %left, %vl
  %done = icmp eq i64 %next, 0
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}
attributes #0 = { "trex-bosc-bound-rvv"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
