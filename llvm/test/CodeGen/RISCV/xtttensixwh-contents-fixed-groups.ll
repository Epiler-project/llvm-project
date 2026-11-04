; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST
;
; Official WH/BH fixed groups have identical numerical models and fields.
; Each function has one real runtime loop, independent contents PHIs, an
; actual zero bypass and all returned roles observed. No source iteration is
; copied. Native old/result constraints and WH cooldown remain separate.

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.rotate4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

; CHECK-LABEL: <copy4_0_loop>:
; CHECK: beqz
; CHECK: ttsfpshft2.copy4 0
; CHECK-NOT: ttsfpshft2.copy4
; CHECK: ttsfpstore l0, 32, 0, 4
; CHECK: ttsfpstore l1, 34, 0, 4
; CHECK: ttsfpstore l2, 36, 0, 4
; CHECK: ttsfpstore l3, 38, 0, 4
; CHECK: ret
; ISEL-LABEL: name: copy4_0_loop
; ISEL: :sfpr = PHI
; ISEL: PseudoTTSFPContentsCopy4
; POST-LABEL: name: copy4_0_loop
; POST: BEQ
; POST: TTSFPSHFT2Copy4 0
; POST-SAME: implicit-def $tt_l0
; POST-SAME: implicit $tt_l0
define void @copy4_0_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %raw0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw0, i32 0, i32 5)
  %raw1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw1, i32 64, i32 5)
  %raw2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw2, i32 128, i32 5)
  %raw3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw3, i32 192, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed0, i32 0, i32 0, i32 4)
  %has.iterations = icmp ugt i32 %limit, 0
  br i1 %has.iterations, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, i32 0)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.stallwait(i32 16384, i32 511)
  call void @llvm.riscv.tt.nop()
  ret void
}

; CHECK-LABEL: <copy4_1_loop>:
; CHECK: beqz
; CHECK: ttsfpshft2.copy4 1
; CHECK-NOT: ttsfpshft2.copy4
; CHECK: ttsfpstore l0, 32, 0, 4
; CHECK: ttsfpstore l1, 34, 0, 4
; CHECK: ttsfpstore l2, 36, 0, 4
; CHECK: ttsfpstore l3, 38, 0, 4
; CHECK: ret
; ISEL-LABEL: name: copy4_1_loop
; ISEL: :sfpr = PHI
; ISEL: PseudoTTSFPContentsCopy4
; POST-LABEL: name: copy4_1_loop
; POST: BEQ
; POST: TTSFPSHFT2Copy4 1
; POST-SAME: implicit-def $tt_l0
; POST-SAME: implicit $tt_l0
define void @copy4_1_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %raw0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw0, i32 0, i32 5)
  %raw1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw1, i32 64, i32 5)
  %raw2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw2, i32 128, i32 5)
  %raw3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw3, i32 192, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed0, i32 0, i32 0, i32 4)
  %has.iterations = icmp ugt i32 %limit, 0
  br i1 %has.iterations, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, i32 1)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.stallwait(i32 16384, i32 511)
  call void @llvm.riscv.tt.nop()
  ret void
}

; CHECK-LABEL: <rotate4_loop>:
; CHECK: beqz
; CHECK: ttsfpshft2.rotate4 l0
; CHECK-NOT: ttsfpshft2.rotate4
; CHECK: ttsfpstore l0, 32, 0, 4
; CHECK: ttsfpstore l1, 34, 0, 4
; CHECK: ttsfpstore l2, 36, 0, 4
; CHECK: ttsfpstore l3, 38, 0, 4
; CHECK: ret
; ISEL-LABEL: name: rotate4_loop
; ISEL: :sfpr = PHI
; ISEL: PseudoTTSFPContentsRotate4
; POST-LABEL: name: rotate4_loop
; POST: BEQ
; POST: TTSFPSHFT2Rotate4
; POST-SAME: implicit-def $tt_l0
; POST-SAME: implicit $tt_l0
define void @rotate4_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %raw0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw0, i32 0, i32 5)
  %raw1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw1, i32 64, i32 5)
  %raw2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw2, i32 128, i32 5)
  %raw3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw3, i32 192, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed0, i32 0, i32 0, i32 4)
  %has.iterations = icmp ugt i32 %limit, 0
  br i1 %has.iterations, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.rotate4(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, <32 x i32> %active0)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.stallwait(i32 16384, i32 511)
  call void @llvm.riscv.tt.nop()
  ret void
}

; CHECK-LABEL: <transpose8_loop>:
; CHECK: beqz
; CHECK: ttsfptransp
; CHECK-NOT: ttsfptransp
; CHECK: ttsfpstore l0, 32, 0, 4
; CHECK: ttsfpstore l1, 34, 0, 4
; CHECK: ttsfpstore l2, 36, 0, 4
; CHECK: ttsfpstore l3, 38, 0, 4
; CHECK: ttsfpstore l4, 40, 0, 4
; CHECK: ttsfpstore l5, 42, 0, 4
; CHECK: ttsfpstore l6, 44, 0, 4
; CHECK: ttsfpstore l7, 46, 0, 4
; CHECK: ret
; ISEL-LABEL: name: transpose8_loop
; ISEL: :sfpr = PHI
; ISEL: PseudoTTSFPContentsTranspose8
; POST-LABEL: name: transpose8_loop
; POST: BEQ
; POST: TTSFPTRANSP
; POST-SAME: implicit-def $tt_l0
; POST-SAME: implicit $tt_l0
define void @transpose8_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %raw0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw0, i32 0, i32 5)
  %raw1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw1, i32 64, i32 5)
  %raw2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw2, i32 128, i32 5)
  %raw3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw3, i32 192, i32 5)
  %raw4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed4 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw4, i32 256, i32 5)
  %raw5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed5 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw5, i32 320, i32 5)
  %raw6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed6 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw6, i32 384, i32 5)
  %raw7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed7 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw7, i32 448, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed0, i32 0, i32 0, i32 4)
  %has.iterations = icmp ugt i32 %limit, 0
  br i1 %has.iterations, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %active4 = phi <32 x i32> [ %seed4, %entry ], [ %updated4, %latch ]
  %active5 = phi <32 x i32> [ %seed5, %entry ], [ %updated5, %latch ]
  %active6 = phi <32 x i32> [ %seed6, %entry ], [ %updated6, %latch ]
  %active7 = phi <32 x i32> [ %seed7, %entry ], [ %updated7, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, <32 x i32> %active4, <32 x i32> %active5, <32 x i32> %active6, <32 x i32> %active7)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  %updated4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 4
  %updated5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 5
  %updated6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 6
  %updated7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 7
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %final4 = phi <32 x i32> [ %seed4, %entry ], [ %updated4, %latch ]
  %final5 = phi <32 x i32> [ %seed5, %entry ], [ %updated5, %latch ]
  %final6 = phi <32 x i32> [ %seed6, %entry ], [ %updated6, %latch ]
  %final7 = phi <32 x i32> [ %seed7, %entry ], [ %updated7, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final4, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final5, i32 42, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final6, i32 44, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final7, i32 46, i32 0, i32 4)
  call void @llvm.riscv.tt.stallwait(i32 16384, i32 511)
  call void @llvm.riscv.tt.nop()
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
