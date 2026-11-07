; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/assumed.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/wide.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/missing-noundef.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/nondominating.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-cycle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/wide.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/missing-noundef.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/nondominating.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/poison-cycle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; The typed stream-bank ABI owns the initialized load domain [0, 64).
; Keep both range and noundef metadata through CodeGenPrepare: assumptions
; alone are temporary, and freeze cannot preserve range without noundef.
; The field receiver still rejects absent/wide/non-dominating domains,
; disjoint-bit overlap, and a masked poison predecessor.
; CHECK-LABEL: bounded_abi:
; CHECK: sw
; CHECK: ret
; POISON: STOREREG field regaddr must be defined and non-poison

;--- valid.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !range !0, !noundef !1
  %bank = freeze i32 %word
  %scaled = shl nuw nsw i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- assumed.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base
  %bank = freeze i32 %word
  %inrange = icmp ult i32 %bank, 64
  call void @llvm.assume(i1 %inrange)
  %scaled = shl nuw nsw i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

;--- wide.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !range !0, !noundef !1
  %bank = freeze i32 %word
  %scaled = shl nuw nsw i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 128}
!1 = !{}

;--- unbounded.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !noundef !1
  %bank = freeze i32 %word
  %scaled = shl nuw nsw i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- missing-noundef.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !range !0
  %bank = freeze i32 %word
  %scaled = shl nuw nsw i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- overlap.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !range !0, !noundef !1
  %bank = freeze i32 %word
  %bit = and i32 %bank, 1
  %scaled = shl i32 %bit, 10
  %regaddr = or disjoint i32 %scaled, 1024
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- poison.ll
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-elf"
@runtime = external global ptr

declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @bounded_abi() "tensix-executor"="trisc1" {
  %base = load ptr, ptr @runtime
  %word = load i32, ptr %base, !range !0, !noundef !1
  %bank = freeze i32 %word
  %overflow = shl nuw i32 %bank, 31
  %masked = and i32 %overflow, 63
  %scaled = shl i32 %masked, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

!0 = !{i32 0, i32 64}
!1 = !{}

;--- nondominating.ll
declare void @llvm.assume(i1 noundef)
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @branch_local_assumption(i32 noundef %bank, i1 noundef %choose) "tensix-executor"="trisc1" {
entry:
  br i1 %choose, label %narrow, label %other
narrow:
  %inrange = icmp ult i32 %bank, 64
  call void @llvm.assume(i1 %inrange)
  br label %join
other:
  br label %join
join:
  %scaled = shl i32 %bank, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  ret void
}

;--- poison-cycle.ll
declare void @llvm.riscv.tt.storereg.port(i32 immarg, i32, i32)
define void @overflowing_cycle() "tensix-executor"="trisc1" {
entry:
  br label %loop
loop:
  %counter = phi i32 [ 0, %entry ], [ %next, %loop ]
  %bounded = and i32 %counter, 63
  %scaled = shl i32 %bounded, 10
  %regaddr = or disjoint i32 %scaled, 65544
  call void @llvm.riscv.tt.storereg.port(i32 0, i32 %regaddr, i32 60)
  %next = add nuw i32 %counter, 1
  br label %loop
}
