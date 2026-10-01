; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o %t.prepare.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.final.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: %python %S/Inputs/tensix-replay-dynamic-dst-observer.py %t.o0.s %t.prepare.mir %t.final.mir 1
; RUN: %python %S/Inputs/tensix-replay-dynamic-dst-observer.py %t.o2.s %t.prepare.mir %t.final.mir 1
;
; Runtime offset capture must survive both record modes. The later real loop
; executes the stored address fields against current Dst/LReg state. No loop
; unrolling, hidden SFPR allocation, or record-time numerical snapshot.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @dynamic_dst_record_and_execute(i10 noundef %input, i10 noundef %output, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %read_offset = zext i10 %input to i32
  %write_offset = zext i10 %output to i32
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.replay.template.begin(i32 701, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %read_offset, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 %write_offset, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 701)
  ; Observe current L0 before any later execute. Record-only preserves C10;
  ; record-and-execute already performed the load from the captured address.
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.replay.template.execute(i32 701)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  ret void
}
