# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/wide.s 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/wide.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/wide.txt 2>&1 | FileCheck %s --check-prefix=RAW

# WH scalar arithmetic addresses 64 GPRs. Its immediate selector is raw
# bit 23, distinct from unused destination bits 22:18.
#--- valid.s
ttadddmareg 1, 2, 3, 0
ttadddmareg 63, 63, 63, 1
ttmuldmareg 3, 2, 1, 0
ttmuldmareg 4, 31, 63, 1
ttdmanop
# CHECK: ttadddmareg 1, 2, 3, 0{{ *}}# encoding: [0x05,0xc2,0x00,0x60]
# CHECK: ttadddmareg 63, 63, 63, 1{{ *}}# encoding: [0xfd,0xff,0x0f,0x62]
# CHECK: ttmuldmareg 3, 2, 1, 0{{ *}}# encoding: [0x0d,0x42,0x00,0x68]
# CHECK: ttmuldmareg 4, 31, 63, 1{{ *}}# encoding: [0x11,0xdf,0x0f,0x6a]
# CHECK: ttdmanop{{ *}}# encoding: [0x01,0x00,0x00,0x80]
# DIS: ttadddmareg 1, 2, 3, 0
# DIS-NEXT: {{.*}}ttadddmareg 63, 63, 63, 1
# DIS-NEXT: {{.*}}ttmuldmareg 3, 2, 1, 0
# DIS-NEXT: {{.*}}ttmuldmareg 4, 31, 63, 1
# DIS-NEXT: {{.*}}ttdmanop

#--- wide.s
ttadddmareg 0, 0, 64, 0
ttmuldmareg 0, 0, 1024, 0
# INVALID: error: Wormhole ADDDMAREG field resultregindex must be in [0, 63]
# INVALID: error: Wormhole MULDMAREG field resultregindex must be in [0, 63]
# BH: ttadddmareg 0, 0, 64, 0{{ *}}# encoding: [0x01,0x00,0x10,0x60]
# BH: ttmuldmareg 0, 0, 1024, 0{{ *}}# encoding: [0x01,0x00,0x00,0x69]

#--- wide.txt
0x01 0x00 0x10 0x60 0x01 0x00 0x00 0x69 0x01 0x00 0x00 0x80
# RAW-COUNT-2: warning: invalid instruction encoding
# RAW: ttdmanop
