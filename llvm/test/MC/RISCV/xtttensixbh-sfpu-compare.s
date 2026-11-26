# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=FEATURE

# SET_CC alone reads both registers, including aliased operands. The explicit
# VD read is restricted to L0..L7 so no backdoor-load configuration is assumed.
# Raw words follow the Blackhole ISA independently of generated encoders:
# SFPGT=0x97, SFPLE=0x96, VC[11:8], VD[7:4], mode[3:0], rotate-left two.
# ENC: ttsfpgt l0, l1, 1 # encoding: [0x06,0x04,0x00,0x5c]
# DIS: 0: {{.*}}ttsfpgt l0, l1, 1
# FEATURE: instruction requires the following: 'XTTTensixBH'
ttsfpgt l0, l1, 1
# ENC: ttsfple l7, c15, 1 # encoding: [0xc6,0x3d,0x00,0x58]
# DIS: 4: {{.*}}ttsfple l7, c15, 1
ttsfple l7, c15, 1
# ENC: ttsfpgt l3, l3, 1 # encoding: [0xc6,0x0c,0x00,0x5c]
# DIS: 8: {{.*}}ttsfpgt l3, l3, 1
ttsfpgt l3, l3, 1
