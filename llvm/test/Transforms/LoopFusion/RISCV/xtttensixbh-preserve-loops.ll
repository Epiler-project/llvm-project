; RUN: opt -S -passes=loop-fusion -loop-fusion-peel-max-count=3 %s | FileCheck %s --check-prefix=KEEP --implicit-check-not=.peel
; RUN: sed 's/"target-features"="+xtttensixbh" "tensix-executor"="trisc1"/"tensix-executor"="ncrisc"/g' %s | opt -S -passes=loop-fusion -loop-fusion-peel-max-count=3 | FileCheck %s --check-prefix=KEEP --implicit-check-not=.peel
; RUN: sed 's/"target-features"="+xtttensixbh" "tensix-executor"="trisc1"/"target-cpu"="generic-rv32"/g' %s | opt -S -passes=loop-fusion -loop-fusion-peel-max-count=3 | FileCheck %s --check-prefix=FUSE
; REQUIRES: riscv-registered-target
;
; Fusion must not peel Tensix executor iterations to equalize trip counts.
; The identical pair on ordinary RISC-V remains eligible for peeling/fusion.
target triple = "riscv32"

; KEEP-LABEL: define void @unequal(
; KEEP: first:
; KEEP: phi i32
; KEEP: store i32
; KEEP: second:
; KEEP: phi i32
; KEEP: store i32
; KEEP: ret void
; FUSE-LABEL: define void @unequal(
; FUSE: first.peel:
; FUSE: store i32
; FUSE: ret void
define void @unequal(ptr noalias %a, ptr noalias %b) #0 {
entry:
  br label %first
first:
  %i = phi i32 [ 0, %entry ], [ %inext, %first.latch ]
  %ap = getelementptr i32, ptr %a, i32 %i
  store i32 %i, ptr %ap, align 4
  br label %first.latch
first.latch:
  %inext = add nuw nsw i32 %i, 1
  %idone = icmp eq i32 %inext, 100
  br i1 %idone, label %second.preheader, label %first
second.preheader:
  br label %second
second:
  %j = phi i32 [ 1, %second.preheader ], [ %jnext, %second.latch ]
  %bp = getelementptr i32, ptr %b, i32 %j
  store i32 %j, ptr %bp, align 4
  br label %second.latch
second.latch:
  %jnext = add nuw nsw i32 %j, 1
  %jdone = icmp eq i32 %jnext, 100
  br i1 %jdone, label %exit, label %second
exit:
  ret void
}
attributes #0 = { "target-features"="+xtttensixbh" "tensix-executor"="trisc1" }
