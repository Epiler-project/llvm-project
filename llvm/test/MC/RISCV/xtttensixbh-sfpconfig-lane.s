# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: not llvm-mc -triple=riscv32 %t/valid.s 2>&1 | FileCheck %s --check-prefix=FEATURE
# ENC: ttsfpconfig.lane 21844, 8 # encoding: [0xe2,0x53,0x55,0x45]
# ENC: ttsfpconfig.lane 21845, 8 # encoding: [0xe2,0x57,0x55,0x45]
# ENC: ttsfpconfig.lane 0, 8 # encoding: [0xe2,0x03,0x00,0x44]
# DIS: ttsfpconfig.lane 21844, 8
# DIS: ttsfpconfig.lane 21845, 8
# DIS: ttsfpconfig.lane 0, 8
# INVALID: SFPCONFIG mode 8 mask must select even lane bits
# INVALID: SFPCONFIGLane mode must be one of {8}
# FEATURE: instruction requires the following: 'XTTTensixBH'

#--- valid.s
ttsfpconfig.lane 21844, 8
ttsfpconfig.lane 21845, 8
ttsfpconfig.lane 0, 8

#--- invalid.s
ttsfpconfig.lane 2, 8
ttsfpconfig.lane 0, 0
