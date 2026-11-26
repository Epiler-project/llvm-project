; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/guarded.ll -o /dev/null
; RUN: sed 's/icmp ult/icmp ule/' %t/guarded.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=DST-POISON
; DST-POISON: bound Dst offset must be defined and non-poison
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/nested.ll -o /dev/null
; RUN: sed 's/icmp ult i32 %%packet/icmp ule i32 %%packet/' %t/nested.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=DST-POISON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/fields.ll -o /dev/null
; RUN: sed 's/i32 192/i32 224/' %t/fields.ll | not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -o /dev/null 2>&1 | FileCheck %s --check-prefix=DST-POISON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/assertion-only.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/undef-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison-masked-input.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
;
; InstCombine rewrites the Unpack face x-end expression (face & 1) * 16 + 15
; to a disjoint OR. Prove disjointness from the operands, then independently
; prove their definedness through the real loop backedge. The flag itself is
; not evidence, and a disjoint mask cannot repair a poisoned operand.
; CHECK-LABEL: unpack_face_ends:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; CHECK-LABEL: disjoint_dynamic_fields:
; CHECK: sw
; CHECK: {{b(ne|lt|ge|eq)}}
; CHECK: ret
; POISON: SETADCXX field x_end2 must be defined and non-poison

;--- valid.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)

define void @unpack_face_ends() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp slt i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %shift = shl i32 %face, 4
  %begin = and i32 %shift, 16
  %end = or disjoint i32 %begin, 15
  call void @llvm.tt.setadcxx.port(i32 0, i32 %begin, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

define void @disjoint_dynamic_fields() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %shift = shl i32 %face, 4
  %lo = and i32 %face, 3
  %hi = and i32 %shift, 48
  %end = or disjoint i32 %hi, %lo
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- overlap.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @overlap() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %bit = and i32 %face, 1
  %end = or disjoint i32 %bit, 1
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- assertion-only.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @assertion_only(i32 noundef %value) "tensix-executor"="trisc0" {
  %bounded = and i32 %value, 31
  %end = or disjoint i32 %bounded, 15
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- guarded.ll
declare void @llvm.tt.bh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
define void @copy_section() "tensix-executor"="trisc1" {
entry:
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %i, 64
  br i1 %active, label %body, label %exit
body:
  %addr = or disjoint i32 %i, 64
  call void @llvm.tt.bh.sfpu.store(i32 0, i32 %addr, i32 7, i32 2)
  %next = add nuw nsw i32 %i, 2
  br label %header
exit:
  ret void
}

;--- nested.ll
declare void @llvm.tt.bh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
define void @copy_faces() "tensix-executor"="trisc1" {
entry:
  br label %faces
faces:
  %face = phi i32 [ 0, %entry ], [ %next.face, %advance ]
  %face.active = icmp ult i32 %face, 4
  br i1 %face.active, label %base, label %exit
base:
  %hi = shl nuw nsw i32 %face, 4
  br label %packets
packets:
  %packet = phi i32 [ 0, %base ], [ %next.packet, %body ]
  %packet.active = icmp ult i32 %packet, 8
  br i1 %packet.active, label %body, label %advance
body:
  %lo = shl nuw nsw i32 %packet, 1
  %row = or disjoint i32 %hi, %lo
  %offset = or disjoint i32 %row, 192
  call void @llvm.tt.bh.sfpu.store(i32 0, i32 %offset, i32 7, i32 2)
  %next.packet = add nuw nsw i32 %packet, 1
  br label %packets
advance:
  %next.face = add nuw nsw i32 %face, 1
  br label %faces
exit:
  ret void
}

;--- poison-input.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @poison_input() "tensix-executor"="trisc0" {
  %bounded = and i32 poison, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- undef-input.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @undef_input(i32 %value) "tensix-executor"="trisc0" {
  %bounded = and i32 %value, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  ret void
}

;--- poison-masked-input.ll
declare void @llvm.tt.setadcxx.port(i32 immarg, i32, i32, i32)
define void @poison_masked_input() "tensix-executor"="trisc0" {
entry:
  br label %header
header:
  %face = phi i32 [ 0, %entry ], [ %next, %body ]
  %active = icmp ult i32 %face, 4
  br i1 %active, label %body, label %exit
body:
  %overflow = shl nuw i32 %face, 31
  %bounded = and i32 %overflow, 16
  %end = or disjoint i32 %bounded, 15
  call void @llvm.tt.setadcxx.port(i32 0, i32 0, i32 %end, i32 1)
  %next = add i32 %face, 1
  br label %header
exit:
  ret void
}

;--- fields.ll
declare void @llvm.tt.bh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
define void @packet_offset(i32 noundef %face, i32 noundef %packet,
                            i1 noundef %page) "tensix-executor"="trisc1" {
entry:
  %face.active = icmp ult i32 %face, 2
  br i1 %face.active, label %packets, label %exit
packets:
  %packet.active = icmp ult i32 %packet, 4
  br i1 %packet.active, label %body, label %exit
body:
  %face.bits = shl nuw nsw i32 %face, 5
  %packet.bits = shl nuw nsw i32 %packet, 2
  %row = or disjoint i32 %face.bits, %packet.bits
  %section = select i1 %page, i32 256, i32 192
  %addr = or disjoint i32 %section, %row
  call void @llvm.tt.bh.sfpu.store(i32 0, i32 %addr, i32 7, i32 2)
  br label %exit
exit:
  ret void
}
