; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/wide.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
;
; Scalar scheduling arithmetic is legal across live bound SFPU state. These
; word-sized min/max operations select ordinary GPR compares/selects, not
; calls, SFPR temporaries or copies. Wide/vector intrinsics are not admitted
; by this rule; they need a separate legalization contract.
;--- valid.ll
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define i32 @scalar_minmax(i32 noundef %a, i32 noundef %b) "tensix-executor"="trisc1" {
; ASM-LABEL: scalar_minmax:
; ASM-NOT: call
; ASM: .word 0xf0002409
; ASM-NOT: call
; ASM: .word 0xc8100001
; ASM-NOT: call
; ASM: ret
; ISEL-LABEL: name: scalar_minmax
; ISEL-NOT: class: sfpr
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NOT: PseudoCALL
; ISEL: TTSFPSTORE $tt_l0, 0, 0, 4,
; ISEL-NOT: PseudoCALL
; ISEL: PseudoRET
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  %smax = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  %smin = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  %umax = call i32 @llvm.umax.i32(i32 %a, i32 %b)
  %umin = call i32 @llvm.umin.i32(i32 %a, i32 %b)
  %signed = xor i32 %smax, %smin
  %unsigned = sub i32 %umax, %umin
  %result = xor i32 %signed, %unsigned
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret i32 %result
}

;--- wide.ll
declare i64 @llvm.smax.i64(i64, i64)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
define i64 @wide(i64 %a, i64 %b) "tensix-executor"="trisc1" {
; REJECT: call has no verified SFPU preservation ABI
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  %result = call i64 @llvm.smax.i64(i64 %a, i64 %b)
  ret i64 %result
}
