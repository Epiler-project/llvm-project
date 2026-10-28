; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %s | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj < %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ

; The formal intrinsic ABI is shared by both models. These canonical Full
; forms invalidate Dst row validity, not numerical tensor payload. The target
; encoder owns use32 at WH raw bit21 versus BH raw bit18 and the distinct
; AddrMod positions. No packed mode7 argument or wrapper call is needed.
declare void @llvm.riscv.tt.zeroacc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @full_fp16() "tensix-executor"="trisc2" {
; WH-LABEL: full_fp16:
; WH: .word 0x40620000
; WH-NOT: call
; WH: ret
; BH-LABEL: full_fp16:
; BH: .word 0x40610000
; BH-NOT: call
; BH: ret
; ISEL-LABEL: name: full_fp16
; ISEL: TTZEROACC 0, 1, 0, 0, 3
; OBJ-LABEL: <full_fp16>:
; OBJ: ttzeroacc 0, 1, 0, 0, 3
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 0, i32 3)
  ret void
}

define void @full_fp32() "tensix-executor"="trisc2" {
; WH-LABEL: full_fp32:
; WH: .word 0x40e20000
; WH-NOT: call
; WH: ret
; BH-LABEL: full_fp32:
; BH: .word 0x40710000
; BH-NOT: call
; BH: ret
; ISEL-LABEL: name: full_fp32
; ISEL: TTZEROACC 0, 1, 0, 1, 3
; OBJ-LABEL: <full_fp32>:
; OBJ: ttzeroacc 0, 1, 0, 1, 3
; OBJ-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.zeroacc(i32 0, i32 1, i32 0, i32 1, i32 3)
  ret void
}
