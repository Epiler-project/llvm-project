; Dedicated SFPR SSA is selected before the physical-ingress boundary. The
; full compiler accepts only physical SFPU bindings; the boundary rejection
; itself is covered by xtttensixbh-sfpu-physical-ingress.mir. Keep this file
; selector-only so retained loops cannot continue into the generic verifier
; after the intentional boundary diagnostic.
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)

; Keeping %three alive across subtraction forces a full-lane copy before the
; tied old-destination operand is overwritten, including inactive lanes.
define void @destructive_later_use() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: destructive_later_use
; ISEL: TTSFPISUB
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %three = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> %zero, i32 3, i32 2)
  %seven = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> %zero, i32 7, i32 2)
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %zero, i32 0, i32 2)
  %difference = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %three, <32 x i32> %seven, i32 6)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %difference)
  call void @llvm.riscv.tt.lreg.write(i32 4, <32 x i32> %three)
  ret void
}

; %before must remain independent of the fixed write and the second snapshot.
define void @fixed_snapshot() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: fixed_snapshot
; ISEL: $tt_l3
  %before = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %zero)
  %after = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  call void @llvm.riscv.tt.lreg.write(i32 4, <32 x i32> %before)
  call void @llvm.riscv.tt.lreg.write(i32 5, <32 x i32> %after)
  ret void
}

define void @retained_sfpu_loop(i32 %limit) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: retained_sfpu_loop
; ISEL: PHI
; ISEL: TTSFPIADD
entry:
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> %zero, i32 1, i32 2)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%zero, %entry], [%sum, %loop]
  %sum = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %value, <32 x i32> %one, i32 4)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %sum)
  ret void
}

; Raw format4 and bounded dynamic scalar offset survive until post-RA expansion.
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

; SRCB mode is an architectural format selector. It reads the selected SFPU
; destination configuration at execution and must survive both static MC
; emission and bounded dynamic instruction-port emission without remapping.
define void @configured_dst_format() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: configured_dst_format
; ISEL: TTSFPLOAD {{.*}}, 0, 7, 0
; ISEL: TTSFPSTORE {{.*}}, 2, 7, 0
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 0, i32 7, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %value, i32 2, i32 7, i32 0)
  ret void
}

define void @dynamic_configured_dst_format(i10 noundef %index) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_configured_dst_format
; ISEL: PseudoTTSFPLOAD {{.*}}, 7, 0
; ISEL: PseudoTTSFPSTORE {{.*}}, 7, 0
  %offset = zext i10 %index to i32
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 %offset, i32 7, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %value, i32 %offset, i32 7, i32 0)
  ret void
}

define void @dynamic_dst(i10 noundef %index) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_dst
; ISEL: PseudoTTSFPLOAD
; ISEL: PseudoTTSFPSTORE
  %offset = zext i10 %index to i32
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 %offset, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  ret void
}

; The address recurrence remains one scalar loop, including at O0. Range and
; definedness must be established without duplicating iterations or trusting
; unchecked nuw/nsw promises.
define void @retained_dst_loop() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: retained_dst_loop
; ISEL: PHI
; ISEL: PseudoTTSFPSTORE
entry:
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %zero, i32 %i, i32 0, i32 4)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}


define void @retained_flagged_dst_loop() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: retained_flagged_dst_loop
; ISEL: PHI
; ISEL: PseudoTTSFPSTORE
entry:
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %offset = shl nuw nsw i32 %i, 1
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %zero, i32 %offset, i32 0, i32 4)
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

define void @dynamic_bounded_safe(i32 noundef %input) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_bounded_safe
; ISEL: PHI
; ISEL: PseudoTTSFPSTORE
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %start = and i32 %input, 15
  br label %loop
loop:
  %count = phi i32 [0, %entry], [%count.next, %loop]
  %i = phi i32 [%start, %entry], [%next, %loop]
  %offset = shl nuw nsw i32 %i, 1
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add nuw nsw i32 %i, 1
  %count.next = add i32 %count, 1
  %more = icmp ult i32 %count.next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
