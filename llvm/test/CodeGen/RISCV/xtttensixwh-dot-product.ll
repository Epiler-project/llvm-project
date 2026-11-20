; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ

declare void @llvm.riscv.tt.dotpv(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @dot() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dot
; ISEL: TTDOTPV 24, 3, 0, 0, 3
; OBJ-LABEL: <dot>:
; OBJ: ttdotpv 24, 3, 0, 0, 3
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.dotpv(i32 24, i32 3, i32 0, i32 0, i32 3)
  ret void
}
