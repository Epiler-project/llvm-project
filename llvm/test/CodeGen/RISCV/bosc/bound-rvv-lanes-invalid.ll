; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/short-write.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/branch.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/backedge.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/ambiguous.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/dynamic.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: read of undefined physical vector register v4
; CHECK-SAME: covering the active setvl lanes

;--- short-write.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @short_write(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- branch.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @branch(ptr %p, i1 %cond) "trex-bosc-bound-rvv"="1" {
entry:
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br i1 %cond, label %short, label %join
short:
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br label %join
join:
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- backedge.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @backedge(ptr %p, i1 %again) "trex-bosc-bound-rvv"="1" {
entry:
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br label %loop
loop:
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
;--- ambiguous.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @ambiguous(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 17)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
;--- dynamic.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
define void @dynamic(ptr %p, i64 %n) "trex-bosc-bound-rvv"="1" {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 %n)
  call void @llvm.riscv.bosc.load(i32 4, ptr %p)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
