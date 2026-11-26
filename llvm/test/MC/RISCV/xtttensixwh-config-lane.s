# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %t/valid.s 2>&1 | FileCheck %s --check-prefix=FEATURE
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=MODE
# ENC: ttsfpconfig.lane 21845, 8{{ *}}# encoding: [0xe2,0x57,0x55,0x45]
# DIS: ttsfpconfig.lane 21845, 8
# FEATURE: error: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
# MODE: error: SFPCONFIGLane mode must be one of {8}
#--- valid.s
ttsfpconfig.lane 21845, 8
#--- invalid.s
ttsfpconfig.lane 21845, 7
