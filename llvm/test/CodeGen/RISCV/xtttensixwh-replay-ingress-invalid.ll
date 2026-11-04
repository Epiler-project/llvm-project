; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/raw.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/port.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/mop.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/legacy-template.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs %t/legacy-end.ll -o /dev/null 2>&1 | FileCheck %s
;
; Receiving allocated contents templates and automatic SFPU compression must
; not open raw/bound controls, instruction-port or MOP ingress on WH.
; Native MC REPLAY is a serialization boundary, not a second LLVM ingress.
; CHECK: Tensix intrinsic is not supported on Wormhole

;--- raw.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @raw_control() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 0)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.replay.port(i32 immarg, i32, i32, i32, i32)
define void @port_control() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.port(i32 0, i32 1, i32 1, i32 1, i32 0)
  ret void
}

;--- mop.ll
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32, i32, i32, i32)
define void @mop_control() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.mop(i32 2, i32 1, i32 1, i32 1, i32 0)
  ret void
}

;--- legacy-template.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
define void @legacy_template() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 41, i32 1, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.end(i32 41)
  call void @llvm.riscv.tt.replay.template.execute(i32 41)
  ret void
}

;--- legacy-end.ll
declare void @llvm.riscv.tt.replay.record.end()
define void @legacy_record_end() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.record.end()
  ret void
}
