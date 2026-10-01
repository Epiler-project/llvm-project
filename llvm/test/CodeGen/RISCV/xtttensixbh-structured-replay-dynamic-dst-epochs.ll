; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: %python %S/Inputs/tensix-replay-dynamic-epoch-observer.py %t.o0.s
; RUN: %python %S/Inputs/tensix-replay-dynamic-epoch-observer.py %t.o2.s
;
; The same lexical record site captures different runtime offsets on each
; visit. A branch may also reuse the previous recording without rerecording.
; Captured fields are not numeric snapshots, and shape equality across a CFG
; backedge does not mean the runtime word bits are equal. Keep the actual CFG.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @rerecord_mode0(i10 noundef %base, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %origin = zext i10 %base to i32
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %record
record:
  %record_iteration = phi i32 [ 0, %entry ], [ %next, %choose ]
  %row = shl i32 %record_iteration, 3
  %logical = add i32 %origin, %row
  %address = and i32 %logical, 1023
  call void @llvm.riscv.tt.replay.template.begin(i32 727, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %address, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 256, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 727)
  br label %execute
execute:
  %iteration = phi i32 [ %record_iteration, %record ]
  call void @llvm.riscv.tt.replay.template.execute(i32 727)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %choose
choose:
  br label %record
exit:
  ret void
}

define void @conditional_record_mode0(i10 noundef %base, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %origin = zext i10 %base to i32
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %record
record:
  %record_iteration = phi i32 [ 0, %entry ], [ %next, %record_edge ]
  %row = shl i32 %record_iteration, 3
  %logical = add i32 %origin, %row
  %address = and i32 %logical, 1023
  call void @llvm.riscv.tt.replay.template.begin(i32 727, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %address, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 256, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 727)
  br label %execute
execute:
  %iteration = phi i32 [ %record_iteration, %record ], [ %next, %skip_edge ]
  call void @llvm.riscv.tt.replay.template.execute(i32 727)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %choose
choose:
  %odd = and i32 %next, 1
  %again = icmp ne i32 %odd, 0
  br i1 %again, label %record_edge, label %skip_edge
record_edge:
  br label %record
skip_edge:
  br label %execute
exit:
  ret void
}

define void @rerecord_mode1(i10 noundef %base, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %origin = zext i10 %base to i32
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %record
record:
  %record_iteration = phi i32 [ 0, %entry ], [ %next, %choose ]
  %row = shl i32 %record_iteration, 3
  %logical = add i32 %origin, %row
  %address = and i32 %logical, 1023
  call void @llvm.riscv.tt.replay.template.begin(i32 727, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %address, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 256, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 727)
  br label %execute
execute:
  %iteration = phi i32 [ %record_iteration, %record ]
  call void @llvm.riscv.tt.replay.template.execute(i32 727)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %choose
choose:
  br label %record
exit:
  ret void
}

define void @conditional_record_mode1(i10 noundef %base, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  %origin = zext i10 %base to i32
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %record
record:
  %record_iteration = phi i32 [ 0, %entry ], [ %next, %record_edge ]
  %row = shl i32 %record_iteration, 3
  %logical = add i32 %origin, %row
  %address = and i32 %logical, 1023
  call void @llvm.riscv.tt.replay.template.begin(i32 727, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %address, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 256, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 727)
  br label %execute
execute:
  %iteration = phi i32 [ %record_iteration, %record ], [ %next, %skip_edge ]
  call void @llvm.riscv.tt.replay.template.execute(i32 727)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %choose
choose:
  %odd = and i32 %next, 1
  %again = icmp ne i32 %odd, 0
  br i1 %again, label %record_edge, label %skip_edge
record_edge:
  br label %record
skip_edge:
  br label %execute
exit:
  ret void
}
