; RUN: opt -S -passes='default<O2>' %s | FileCheck %s
; RUN: opt -S -passes='default<O3>' %s | FileCheck %s
; RUN: opt -S -passes='loop-unroll' -unroll-count=4 -unroll-threshold=100000 -unroll-allow-partial -unroll-runtime %s | FileCheck %s
; RUN: opt -S -passes='loop-unroll' -unroll-force-peel-count=2 -unroll-threshold=100000 %s | FileCheck %s
; REQUIRES: riscv-registered-target
;
; The WH feature itself preserves source iterations, including scalar traversal
; without Tensix intrinsics or a tensix-executor attribute. The volatile store
; makes full/partial unrolling and peeling observable. The metadata deliberately
; requests unrolling so an instruction merely discouraging it cannot prove this.

target triple = "riscv32"

; CHECK-LABEL: define void @wormhole_scalar(
; CHECK-NOT: store volatile
; CHECK: phi i32
; CHECK-NOT: store volatile
; CHECK: store volatile i32
; CHECK-NOT: store volatile
; CHECK: br i1
; CHECK-NOT: store volatile
; CHECK: ret void
define void @wormhole_scalar(ptr %p) #0 {
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

attributes #0 = { "target-features"="+xtttensixwh" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 4}
