; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
;
; Zero and constant fields fold into the instruction word before selection.
; Each call still performs exactly one instruction-port store, including when
; all fields are zero, and runtime fields contribute their exact bits.

declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)

define void @all_zero_fields() #0 {
; CHECK-LABEL: all_zero_fields:
; Independent encoding: opcode 0x45, all logical fields zero => 0x45000000.
; CHECK-DAG: lui [[WORD:[a-z0-9]+]], 282624
; CHECK-DAG: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

define void @nonzero_constant_field() #0 {
; CHECK-LABEL: nonzero_constant_field:
; ResultHalfReg occupies bits 0..6: the store is exactly 0x45000018.
; CHECK: lui [[WORD:[a-z0-9]+]], 282624
; CHECK-NEXT: addi [[WORD]], [[WORD]], 24
; CHECK: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 0, i32 0)
  ret void
}

define void @runtime_field(i32 noundef %input) #0 {
; CHECK-LABEL: runtime_field:
; Bits 8..21 carry the runtime payload: 0x45000000 | (payload << 8).
; CHECK-DAG: lui {{[a-z0-9]+}}, 282624
; CHECK-DAG: slli [[HI:[a-z0-9]+]], a0, 18
; CHECK-DAG: srli {{[a-z0-9]+}}, [[HI]], 10
; CHECK: or [[WORD:[a-z0-9]+]], {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  %payload = and i32 %input, 16383
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 0, i32 0, i32 %payload, i32 0)
  ret void
}

attributes #0 = { "tensix-executor"="trisc0" }
