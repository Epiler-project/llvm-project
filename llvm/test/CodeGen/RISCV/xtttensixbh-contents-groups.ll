; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=finalize-isel -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ISEL

declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.rotate4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

; Every input is an actual independent contents definition. The instruction
; updates both architectural groups; L4..L7 are numerical results, not scratch.
; CHECK-LABEL: <transpose_both_groups>:
; CHECK-DAG: ttsfpmov.all l0, c9
; CHECK-DAG: ttsfpmov.all l1, c10
; CHECK-DAG: ttsfpmov.all l2, c15
; CHECK-DAG: ttsfpmov.all l3, c8
; CHECK-DAG: ttsfpmov.all l4, c9
; CHECK-DAG: ttsfpmov.all l5, c10
; CHECK-DAG: ttsfpmov.all l6, c15
; CHECK-DAG: ttsfpmov.all l7, c8
; CHECK: ttsfptransp
; CHECK: ttsfpstore l0,
; CHECK: ttsfpstore l1,
; CHECK: ttsfpstore l2,
; CHECK: ttsfpstore l3,
; CHECK: ttsfpstore l4,
; CHECK: ttsfpstore l5,
; CHECK: ttsfpstore l6,
; CHECK: ttsfpstore l7,
; ISEL-LABEL: name: transpose_both_groups
; ISEL: class: sfpr0
; ISEL: class: sfpr7
; ISEL: PseudoTTSFPContentsTranspose8
define void @transpose_both_groups() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %e = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %f = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %g = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %h = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d, <32 x i32> %e, <32 x i32> %f, <32 x i32> %g, <32 x i32> %h)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  %r4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 4
  %r5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 5
  %r6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 6
  %r7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 7
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 10, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 12, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 14, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <copy_and_rotate>:
; CHECK: ttsfpshft2.copy4 0
; CHECK: ttsfpshft2.copy4 1
; CHECK: ttsfpshft2.rotate4
; CHECK: ttsfpstore l0,
; CHECK: ttsfpstore l1,
; CHECK: ttsfpstore l2,
; CHECK: ttsfpstore l3,
; ISEL-LABEL: name: copy_and_rotate
; ISEL: PseudoTTSFPContentsCopy4
; ISEL: PseudoTTSFPContentsRotate4
define void @copy_and_rotate() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %first = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d, i32 0)
  %f0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %first, 0
  %f1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %first, 1
  %f2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %first, 2
  %f3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %first, 3
  %second = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %f0, <32 x i32> %f1, <32 x i32> %f2, <32 x i32> %f3, i32 1)
  %s0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %second, 0
  %s1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %second, 1
  %s2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %second, 2
  %s3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %second, 3
  %third = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.rotate4(<32 x i32> %s0, <32 x i32> %s1, <32 x i32> %s2, <32 x i32> %s3, <32 x i32> %s0)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %third, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %third, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %third, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %third, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}

; Group contents cross the backedge as four independent numerical PHIs. There
; is one hardware group instruction in the loop, regardless of runtime trip
; count. Singleton constraints may coalesce these values without repair COPY.
; CHECK-LABEL: <copy_loop>:
; CHECK: ttsfpshft2.copy4 1
; CHECK-NOT: ttsfpshft2.copy4
; CHECK: ttsfpstore l0,
; CHECK: ttsfpstore l1,
; CHECK: ttsfpstore l2,
; CHECK: ttsfpstore l3,
; ISEL-LABEL: name: copy_loop
; ISEL: PHI
; ISEL: PseudoTTSFPContentsCopy4
define void @copy_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %v0 = phi <32 x i32> [ %a, %entry ], [ %r0, %loop ]
  %v1 = phi <32 x i32> [ %b, %entry ], [ %r1, %loop ]
  %v2 = phi <32 x i32> [ %c, %entry ], [ %r2, %loop ]
  %v3 = phi <32 x i32> [ %d, %entry ], [ %r3, %loop ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v0, <32 x i32> %v1, <32 x i32> %v2, <32 x i32> %v3, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  %next = add nuw i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit

exit:
  %final0 = phi <32 x i32> [ %a, %entry ], [ %r0, %loop ]
  %final1 = phi <32 x i32> [ %b, %entry ], [ %r1, %loop ]
  %final2 = phi <32 x i32> [ %c, %entry ], [ %r2, %loop ]
  %final3 = phi <32 x i32> [ %d, %entry ], [ %r3, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 6, i32 0, i32 4)
  ret void
}
