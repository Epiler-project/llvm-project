; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %S/xtttensixbh-contents-creg.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %S/xtttensixbh-contents-creg.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %S/xtttensixbh-contents-creg.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %S/xtttensixbh-contents-creg.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %S/xtttensixbh-contents-creg.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %S/xtttensixbh-contents-creg.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %S/xtttensixbh-contents-creg.ll -o - | FileCheck %S/xtttensixbh-contents-creg.ll --check-prefix=ISEL

; The same C11..C14 snapshot and initialization semantics apply on Wormhole.
; Reuse the source fixture; other contents groups retain their own admission.
