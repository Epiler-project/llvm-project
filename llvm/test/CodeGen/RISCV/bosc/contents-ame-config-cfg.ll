; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/join-successor.ll -o %t/join-successor-O0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/join-successor.ll -o %t/join-successor-O2.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/nested-join-successor.ll -o %t/nested-join-successor-O0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/nested-join-successor.ll -o %t/nested-join-successor-O2.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/nested-config-loops.ll -o %t/nested-config-loops-O0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/nested-config-loops.ll -o %t/nested-config-loops-O2.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/preserving-config-loop.ll -o %t/preserving-config-loop-O0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/preserving-config-loop.ll -o %t/preserving-config-loop-O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/negative-clobber.ll -o %t/negative-clobber-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/negative-clobber.ll -o %t/negative-clobber-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/negative-edge-mismatch.ll -o %t/negative-edge-mismatch-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/negative-edge-mismatch.ll -o %t/negative-edge-mismatch-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/negative-unknown-entry.ll -o %t/negative-unknown-entry-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/negative-unknown-entry.ll -o %t/negative-unknown-entry-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/negative-loop-clobber.ll -o %t/negative-loop-clobber-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/negative-loop-clobber.ll -o %t/negative-loop-clobber-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/negative-nested-row.ll -o %t/negative-nested-row-O0.o 2>&1 | FileCheck %s --check-prefix=ROW
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/negative-nested-row.ll -o %t/negative-nested-row-O2.o 2>&1 | FileCheck %s --check-prefix=ROW

; A configuration association must survive successor blocks, nested joins,
; and dynamic nested/preserving loops. Validate every actual predecessor edge;
; never accept a numeric substitute, a later clobber, or mismatched row PHIs.
; The loops retain their CFG and explicitly disable unrolling.
; CONFIG: AME contents require the actual reaching M/N/K configuration
; ROW: AME contents stride does not cover its configured row

;--- join-successor.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- nested-join-successor.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br i1 %work, label %left2, label %right2
left2:
  br label %join2
right2:
  %mr2 = call i64 @llvm.riscv.bosc.ame.setm(i64 4)
  %nr2 = call i64 @llvm.riscv.bosc.ame.setn(i64 6)
  %kr2 = call i64 @llvm.riscv.bosc.ame.setk(i64 8)
  br label %join2
join2:
  %m2 = phi i64 [ %m, %left2 ], [ %mr2, %right2 ]
  %n2 = phi i64 [ %n, %left2 ], [ %nr2, %right2 ]
  %k2 = phi i64 [ %k, %left2 ], [ %kr2, %right2 ]
  %stride2 = phi i64 [ %stride, %left2 ], [ 24, %right2 ]
  br i1 %choose, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride2, i64 %m2, i64 %n2, i64 %k2)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride2, i64 %m2, i64 %n2, i64 %k2)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- nested-config-loops.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
define void @nested_config_loops(ptr %a, ptr %b, ptr %c, ptr %out, i64 %outer_limit, i64 %inner_limit) #0 {
entry:
  %m0 = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %n0 = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %k0 = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %inext, %outer_latch ]
  %m = phi i64 [ %m0, %entry ], [ %mnext, %outer_latch ]
  %n = phi i64 [ %n0, %entry ], [ %nnext, %outer_latch ]
  %k = phi i64 [ %k0, %entry ], [ %knext, %outer_latch ]
  %stride = phi i64 [ 28, %entry ], [ 20, %outer_latch ]
  %astride = phi i64 [ 4, %entry ], [ 6, %outer_latch ]
  %bstride = phi i64 [ 7, %entry ], [ 5, %outer_latch ]
  %outer_cond = icmp ult i64 %i, %outer_limit
  br i1 %outer_cond, label %outer_body, label %exit
outer_body:
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 %astride, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 %bstride, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %inner
inner:
  %j = phi i64 [ 0, %outer_body ], [ %jnext, %inner_body ]
  %acc = phi target("riscv.bosc.ame.acc") [ %cv, %outer_body ], [ %next_acc, %inner_body ]
  %inner_cond = icmp ult i64 %j, %inner_limit
  br i1 %inner_cond, label %inner_body, label %inner_exit
inner_body:
  %next_acc = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %acc, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  %jnext = add i64 %j, 1
  br label %inner, !llvm.loop !0
inner_exit:
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %acc, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %outer_latch
outer_latch:
  %mnext = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nnext = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %knext = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %inext = add i64 %i, 1
  br label %outer, !llvm.loop !2
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}

;--- preserving-config-loop.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @preserving_config_loop(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br label %loop
loop:
  %mcycle = phi i64 [ %m, %join ], [ %mcycle, %body ]
  %ncycle = phi i64 [ %n, %join ], [ %ncycle, %body ]
  %kcycle = phi i64 [ %k, %join ], [ %kcycle, %body ]
  %stridecycle = phi i64 [ %stride, %join ], [ %stridecycle, %body ]
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stridecycle, i64 %mcycle, i64 %ncycle, i64 %kcycle)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stridecycle, i64 %mcycle, i64 %ncycle, i64 %kcycle)
  br label %loop, !llvm.loop !0
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- negative-clobber.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %later_m = call i64 @llvm.riscv.bosc.ame.setm(i64 1)
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- negative-edge-mismatch.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr_old = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 1)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr_old, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- negative-unknown-entry.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ 6, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- negative-loop-clobber.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @clobbered_config_loop(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  br label %loop
loop:
  %mcycle = phi i64 [ %m, %join ], [ %mcycle, %body ]
  %ncycle = phi i64 [ %n, %join ], [ %ncycle, %body ]
  %kcycle = phi i64 [ %k, %join ], [ %kcycle, %body ]
  %stridecycle = phi i64 [ %stride, %join ], [ %stridecycle, %body ]
  br i1 %work, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stridecycle, i64 %mcycle, i64 %ncycle, i64 %kcycle)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stridecycle, i64 %mcycle, i64 %ncycle, i64 %kcycle)
  %clobber = call i64 @llvm.riscv.bosc.ame.setm(i64 1)
  br label %loop, !llvm.loop !0
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- negative-nested-row.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @joined_config(ptr %c, ptr %out, i1 %choose, i1 %work) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 20, %left ], [ 28, %right ]
  br i1 %work, label %left2, label %right2
left2:
  br label %join2
right2:
  %mr2 = call i64 @llvm.riscv.bosc.ame.setm(i64 4)
  %nr2 = call i64 @llvm.riscv.bosc.ame.setn(i64 6)
  %kr2 = call i64 @llvm.riscv.bosc.ame.setk(i64 8)
  br label %join2
join2:
  %m2 = phi i64 [ %m, %left2 ], [ %mr2, %right2 ]
  %n2 = phi i64 [ %n, %left2 ], [ %nr2, %right2 ]
  %k2 = phi i64 [ %k, %left2 ], [ %kr2, %right2 ]
  %stride2 = phi i64 [ %stride, %left2 ], [ 24, %right2 ]
  br i1 %choose, label %body, label %exit
body:
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride2, i64 %m2, i64 %n2, i64 %k2)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %cv, ptr %out, i64 %stride2, i64 %m2, i64 %n2, i64 %k2)
  br label %exit
exit:
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
