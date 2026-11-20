; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ

declare void @llvm.riscv.tt.mova2d(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.movb2d(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.movd2a(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.movd2b(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @matrix_moves() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: matrix_moves
; ISEL: TTMOVA2D 13, 2, 1, 3, 1
; ISEL-NEXT: TTMOVB2D 13, 5, 1, 3, 1
; ISEL-NEXT: TTMOVD2A 13, 2, 1, 3, 1
; ISEL-NEXT: TTMOVD2B 13, 2, 1, 3, 1
; OBJ-LABEL: <matrix_moves>:
; OBJ: ttmova2d 13, 2, 1, 3, 1
; OBJ-NEXT: {{.*}}ttmovb2d 13, 5, 1, 3, 1
; OBJ-NEXT: {{.*}}ttmovd2a 13, 2, 1, 3, 1
; OBJ-NEXT: {{.*}}ttmovd2b 13, 2, 1, 3, 1
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.mova2d(i32 13, i32 2, i32 1, i32 3, i32 1)
  call void @llvm.riscv.tt.movb2d(i32 13, i32 5, i32 1, i32 3, i32 1)
  call void @llvm.riscv.tt.movd2a(i32 13, i32 2, i32 1, i32 3, i32 1)
  call void @llvm.riscv.tt.movd2b(i32 13, i32 2, i32 1, i32 3, i32 1)
  ret void
}
