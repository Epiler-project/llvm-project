; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov

; A complete immediate writes every result bit but still observes actual CC
; and ROW_MASK. Its native form has no fabricated old operand or tie.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

; ISEL-LABEL: name: full_immediate_mode_0
; ISEL: :sfpr = TTSFPLOADIFull 33059, 0, implicit
; OBJ-LABEL: <full_immediate_mode_0>:
; OBJ: ttsfploadi [[RESULT0:l[0-7]]], 33059, 0
; OBJ-NOT: ttsfploadi
; OBJ: ttsfpstore [[RESULT0]], 0, 0, 3
define void @full_immediate_mode_0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 33059, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

; ISEL-LABEL: name: full_immediate_mode_1
; ISEL: :sfpr = TTSFPLOADIFull 33059, 1, implicit
; OBJ-LABEL: <full_immediate_mode_1>:
; OBJ: ttsfploadi [[RESULT1:l[0-7]]], 33059, 1
; OBJ-NOT: ttsfploadi
; OBJ: ttsfpstore [[RESULT1]], 0, 0, 3
define void @full_immediate_mode_1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 33059, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

; ISEL-LABEL: name: full_immediate_mode_2
; ISEL: :sfpr = TTSFPLOADIFull 33059, 2, implicit
; OBJ-LABEL: <full_immediate_mode_2>:
; OBJ: ttsfploadi [[RESULT2:l[0-7]]], 33059, 2
; OBJ-NOT: ttsfploadi
; OBJ: ttsfpstore [[RESULT2]], 0, 0, 3
define void @full_immediate_mode_2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 33059, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

; ISEL-LABEL: name: full_immediate_mode_4
; ISEL: :sfpr = TTSFPLOADIFull 33059, 4, implicit
; OBJ-LABEL: <full_immediate_mode_4>:
; OBJ: ttsfploadi [[RESULT4:l[0-7]]], 33059, 4
; OBJ-NOT: ttsfploadi
; OBJ: ttsfpstore [[RESULT4]], 0, 0, 3
define void @full_immediate_mode_4() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 33059, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 3)
  ret void
}

; ISEL-LABEL: name: partial_immediate_keeps_exact_old
; ISEL: [[FULL:%[0-9]+]]:sfpr = TTSFPLOADIFull 123, 0, implicit
; ISEL: :sfpr = TTSFPLOADI [[FULL]], 456, 8, implicit
; OBJ-LABEL: <partial_immediate_keeps_exact_old>:
; OBJ: ttsfploadi [[PARTIAL:l[0-7]]], 123, 0
; OBJ: ttsfploadi [[PARTIAL]], 456, 8
; OBJ: ttsfpstore [[PARTIAL]], 4, 0, 3
define void @partial_immediate_keeps_exact_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %full = call <32 x i32> @llvm.riscv.tt.contents.load.immediate.full(i32 123, i32 0)
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %full, i32 456, i32 8)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %partial, i32 4, i32 0, i32 3)
  ret void
}
