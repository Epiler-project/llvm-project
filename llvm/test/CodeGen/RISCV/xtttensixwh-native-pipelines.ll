; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ

declare void @llvm.riscv.tt.mvmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.unpacr(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.zerosrc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

; ISEL-LABEL: name: matmul
; ISEL: TTMVMUL 0, 1, 0, 0
; ISEL: TTMVMUL 16, 2, 1, 2
; ASM-LABEL: matmul:
; ASM: .word 0x98020000
; ASM: .word 0x9a240040
; OBJ-LABEL: <matmul>:
; OBJ: ttmvmul 0, 1, 0, 0
; OBJ: ttmvmul 16, 2, 1, 2
define void @matmul() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.mvmul(i32 0, i32 1, i32 0, i32 0)
  call void @llvm.riscv.tt.mvmul(i32 16, i32 2, i32 1, i32 2)
  ret void
}

; ISEL-LABEL: name: unpack_ab
; ISEL: TTUNPACR 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
; ISEL: TTUNPACR 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1
; ASM-LABEL: unpack_ab:
; ASM: .word 0x08000305
; ASM: .word 0x0a220205
; OBJ-LABEL: <unpack_ab>:
; OBJ: ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
; OBJ: ttunpacr 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1
define void @unpack_ab() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 17, i32 1)
  ret void
}

; ISEL-LABEL: name: zero_source
; ISEL: TTZEROSRC 3, 1, 0, 0
; ISEL: TTZEROSRC 1, 0, 1, 1
; ASM-LABEL: zero_source:
; ASM: .word 0x4400001c
; ASM: .word 0x44000064
; OBJ-LABEL: <zero_source>:
; OBJ: ttzerosrc 3, 1, 0, 0
; OBJ: ttzerosrc 1, 0, 1, 1
define void @zero_source() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.zerosrc(i32 3, i32 1, i32 0, i32 0)
  call void @llvm.riscv.tt.zerosrc(i32 1, i32 0, i32 1, i32 1)
  ret void
}
