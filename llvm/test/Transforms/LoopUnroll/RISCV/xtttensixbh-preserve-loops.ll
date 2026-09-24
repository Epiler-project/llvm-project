; RUN: opt -S -passes='default<O2>' %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='default<O3>' %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='loop-unroll' %s | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='loop-unroll' -unroll-count=4 -unroll-threshold=100000 -unroll-allow-partial -unroll-runtime %s | FileCheck %s
; RUN: opt -S -passes='loop-unroll' -unroll-force-peel-count=2 -unroll-threshold=100000 %s | FileCheck %s
; RUN: sed 's/"target-features"="+xtttensixbh" "tensix-executor"="trisc1"/"tensix-executor"="ncrisc"/g' %s > %t.ncrisc.ll
; RUN: opt -S -passes='default<O2>' %t.ncrisc.ll | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='default<O3>' %t.ncrisc.ll | FileCheck %s --check-prefixes=CHECK,DEFAULT
; RUN: opt -S -passes='loop-unroll' -unroll-count=4 -unroll-threshold=100000 -unroll-allow-partial -unroll-runtime %t.ncrisc.ll | FileCheck %s
; RUN: opt -S -passes='loop-unroll' -unroll-force-peel-count=2 -unroll-threshold=100000 %t.ncrisc.ll | FileCheck %s
; REQUIRES: riscv-registered-target
;
; Tensix source iterations must survive scalar optimization. Volatile stores
; make duplication observable in IR without relying on a numerical oracle or
; SFPU calls that happen to discourage unrolling. The ordinary function uses
; the same loop and retains the normal RISC-V optimization policy.
; NCRISC cannot issue Tensix instructions and therefore has no +xtttensixbh,
; but its scalar traversal obeys the same executor iteration contract.

target triple = "riscv32"

; CHECK-LABEL: define void @hinted(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @hinted(ptr %p) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}

; CHECK-LABEL: define void @size_hinted(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @size_hinted(ptr %p) #1 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}

; CHECK-LABEL: define void @enabled(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @enabled(ptr %p) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop, !llvm.loop !2
exit:
  ret void
}

; CHECK-LABEL: define void @fixed(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @fixed(ptr %p) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK-LABEL: define void @large_fixed(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @large_fixed(ptr %p) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4096
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; CHECK-LABEL: define void @runtime(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @runtime(ptr %p, i32 %n) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

; DEFAULT-LABEL: define void @ordinary(
; DEFAULT-NOT: phi i32
; DEFAULT: store volatile i32 0,
; DEFAULT: store volatile i32 1,
; DEFAULT: store volatile i32 2,
; DEFAULT: store volatile i32 3,
; DEFAULT: ret void
define void @ordinary(ptr %p) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  store volatile i32 %i, ptr %p, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

attributes #0 = { "target-features"="+xtttensixbh" "tensix-executor"="trisc1" }

attributes #1 = { optsize "target-features"="+xtttensixbh" "tensix-executor"="trisc1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 4}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.enable"}
