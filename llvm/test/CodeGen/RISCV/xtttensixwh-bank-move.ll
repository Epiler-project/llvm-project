; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ

declare void @llvm.riscv.tt.movb2a(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @bank_move() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bank_move
; ISEL: TTMOVB2A 5, 2, 3, 7
; OBJ-LABEL: <bank_move>:
; OBJ: ttmovb2a 5, 2, 3, 7
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.movb2a(i32 5, i32 2, i32 3, i32 7)
  ret void
}
