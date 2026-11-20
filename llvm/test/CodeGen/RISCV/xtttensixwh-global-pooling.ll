; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ

declare void @llvm.riscv.tt.gapool(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.gmpool(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @global_pool() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: global_pool
; ISEL: TTGAPOOL 24, 0, 3, 1, 0
; ISEL-NEXT: TTGMPOOL 28, 1, 2, 1, 3
; OBJ-LABEL: <global_pool>:
; OBJ: ttgapool 24, 0, 3, 1, 0
; OBJ-NEXT: {{.*}}ttgmpool 28, 1, 2, 1, 3
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.gapool(i32 24, i32 0, i32 3, i32 1, i32 0)
  call void @llvm.riscv.tt.gmpool(i32 28, i32 1, i32 2, i32 1, i32 3)
  ret void
}
