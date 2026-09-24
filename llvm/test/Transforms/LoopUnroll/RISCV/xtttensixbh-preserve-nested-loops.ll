; RUN: opt -S -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -unroll-remainder %s | FileCheck %s
; RUN: opt -S -passes=loop-unroll-and-jam %s | FileCheck %s
; RUN: sed 's/"target-features"="+xtttensixbh" "tensix-executor"="trisc1"/"tensix-executor"="ncrisc"/g' %s > %t.ncrisc.ll
; RUN: opt -S -passes=loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -unroll-remainder %t.ncrisc.ll | FileCheck %s
; RUN: opt -S -passes=loop-unroll-and-jam %t.ncrisc.ll | FileCheck %s
; REQUIRES: riscv-registered-target
; The same nested sum is eligible for unroll-and-jam on ordinary RISC-V.
; A Tensix target contract preserves both loops even with an explicit hint.
target triple = "riscv32"

; CHECK-LABEL: define void @tensix(
; CHECK-NOT: load i32
; CHECK-NOT: store i32
; CHECK: for.outer:
; CHECK: %i = phi i32
; CHECK-NOT: load i32
; CHECK-NOT: store i32
; CHECK: for.inner:
; CHECK: %j = phi i32
; CHECK: load i32
; CHECK-NOT: load i32
; CHECK: br i1
; CHECK: store i32
; CHECK-NOT: store i32
; CHECK: br i1
; CHECK-NOT: load i32
; CHECK-NOT: store i32
; CHECK: ret void
define void @tensix(i32 %N, i32 %M, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) #0 {
entry:
  %cmp = icmp ne i32 %M, 0
  %cmpJ = icmp ne i32 %N, 0
  %or.cond = and i1 %cmp, %cmpJ
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %M
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %N
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

; CHECK-LABEL: define void @ordinary(
; CHECK-COUNT-4: load i32
; CHECK-COUNT-4: store i32
; CHECK: ret void
define void @ordinary(i32 %N, i32 %M, ptr noalias nocapture %A, ptr noalias nocapture readonly %B) {
entry:
  %cmp = icmp ne i32 %M, 0
  %cmpJ = icmp ne i32 %N, 0
  %or.cond = and i1 %cmp, %cmpJ
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i = phi i32 [ %add8, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j = phi i32 [ 0, %for.outer ], [ %inc, %for.inner ]
  %sum = phi i32 [ 0, %for.outer ], [ %add, %for.inner ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %j
  %0 = load i32, ptr %arrayidx, align 4
  %add = add i32 %0, %sum
  %inc = add nuw i32 %j, 1
  %exitcond = icmp eq i32 %inc, %M
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.lcssa = phi i32 [ %add, %for.inner ]
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %add.lcssa, ptr %arrayidx6, align 4
  %add8 = add nuw i32 %i, 1
  %exitcond25 = icmp eq i32 %add8, %N
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

attributes #0 = { "target-features"="+xtttensixbh" "tensix-executor"="trisc1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll_and_jam.count", i32 4}
