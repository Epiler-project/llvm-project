; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=UNOPT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
;
; X0 contributes zero to every encoded field. Omitting its scalar shift/OR
; must retain the same instruction-port store, including when all fields are
; zero. Nonzero constants and runtime fields still contribute their exact bits.
; These checks cover MC expansion, not deletion of a Tensix issue or a loop.
; O0 may materialize zero in an ordinary GPR. Keep that code valid; only the
; architectural X0 is eligible, without inferring contents of other registers.

declare void @llvm.riscv.tt.setdmareg.port(i32 immarg, i32, i32, i32, i32)

define void @all_zero_fields() #0 {
; UNOPT-LABEL: all_zero_fields:
; UNOPT: lui [[WORD:[a-z0-9]+]], 282624
; UNOPT: lui [[PORT:[a-z0-9]+]], 1048128
; UNOPT-NEXT: sw [[WORD]], 0([[PORT]])
; UNOPT-NOT: sw
; UNOPT: ret
; CHECK-LABEL: all_zero_fields:
; CHECK: lui [[WORD:[a-z0-9]+]], 282624
; CHECK-NEXT: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  ; Independent encoding: opcode 0x45, all logical fields zero => 0x45000000.
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

define void @nonzero_constant_field() #0 {
; UNOPT-LABEL: nonzero_constant_field:
; UNOPT: lui [[WORD:[a-z0-9]+]], 282624
; UNOPT: lui [[PORT:[a-z0-9]+]], 1048128
; UNOPT-NEXT: sw [[WORD]], 0([[PORT]])
; UNOPT-NOT: sw
; UNOPT: ret
; CHECK-LABEL: nonzero_constant_field:
; CHECK: li [[HALF:[a-z0-9]+]], 24
; CHECK: lui [[WORD:[a-z0-9]+]], 282624
; CHECK-NEXT: or [[WORD]], [[WORD]], [[HALF]]
; CHECK-NEXT: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  ; ResultHalfReg occupies bits 0..6: the store remains exactly 0x45000018.
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 24, i32 0, i32 0, i32 0)
  ret void
}

define void @runtime_field(i32 noundef %input) #0 {
; UNOPT-LABEL: runtime_field:
; UNOPT: lui [[WORD:[a-z0-9]+]], 282624
; UNOPT: lui [[PORT:[a-z0-9]+]], 1048128
; UNOPT-NEXT: sw [[WORD]], 0([[PORT]])
; UNOPT-NOT: sw
; UNOPT: ret
; CHECK-LABEL: runtime_field:
; CHECK: lui [[WORD:[a-z0-9]+]], 282624
; CHECK-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, 8
; CHECK-NEXT: or [[WORD]], [[WORD]], [[FIELD]]
; CHECK-NEXT: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[PORT]])
; CHECK-NOT: sw
; CHECK: ret
  %payload = and i32 %input, 16383
  ; Bits 8..21 carry the actual runtime payload: 0x45000000 | (payload << 8).
  call void @llvm.riscv.tt.setdmareg.port(i32 0, i32 0, i32 0, i32 %payload, i32 0)
  ret void
}

attributes #0 = { "tensix-executor"="trisc0" }
