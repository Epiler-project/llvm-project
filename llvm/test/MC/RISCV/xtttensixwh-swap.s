# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# Official SFPSWAP: raw0x92 | VC[11:8] | VD[7:4] | Mod1[3:0], rotated left2.
ttsfpswap l0, l1, 0
# ENC: ttsfpswap l0, l1, 0 # encoding: [0x02,0x04,0x00,0x48]
# DIS: 48000402{{[ \t]+}}ttsfpswap l0, l1, 0
ttsfpswap l0, l1, 1
# ENC: ttsfpswap l0, l1, 1 # encoding: [0x06,0x04,0x00,0x48]
# DIS: 48000406{{[ \t]+}}ttsfpswap l0, l1, 1
ttsfpswap l0, l1, 2
# ENC: ttsfpswap l0, l1, 2 # encoding: [0x0a,0x04,0x00,0x48]
# DIS: 4800040a{{[ \t]+}}ttsfpswap l0, l1, 2
ttsfpswap l0, l1, 3
# ENC: ttsfpswap l0, l1, 3 # encoding: [0x0e,0x04,0x00,0x48]
# DIS: 4800040e{{[ \t]+}}ttsfpswap l0, l1, 3
ttsfpswap l0, l1, 4
# ENC: ttsfpswap l0, l1, 4 # encoding: [0x12,0x04,0x00,0x48]
# DIS: 48000412{{[ \t]+}}ttsfpswap l0, l1, 4
ttsfpswap l0, l1, 5
# ENC: ttsfpswap l0, l1, 5 # encoding: [0x16,0x04,0x00,0x48]
# DIS: 48000416{{[ \t]+}}ttsfpswap l0, l1, 5
ttsfpswap l0, l1, 6
# ENC: ttsfpswap l0, l1, 6 # encoding: [0x1a,0x04,0x00,0x48]
# DIS: 4800041a{{[ \t]+}}ttsfpswap l0, l1, 6
ttsfpswap l0, l1, 7
# ENC: ttsfpswap l0, l1, 7 # encoding: [0x1e,0x04,0x00,0x48]
# DIS: 4800041e{{[ \t]+}}ttsfpswap l0, l1, 7
ttsfpswap l0, l1, 8
# ENC: ttsfpswap l0, l1, 8 # encoding: [0x22,0x04,0x00,0x48]
# DIS: 48000422{{[ \t]+}}ttsfpswap l0, l1, 8
ttsfpswap l0, l1, 9
# ENC: ttsfpswap l0, l1, 9 # encoding: [0x26,0x04,0x00,0x48]
# DIS: 48000426{{[ \t]+}}ttsfpswap l0, l1, 9
ttsfpswap l2, l3, 9
# ENC: ttsfpswap l2, l3, 9 # encoding: [0xa6,0x0c,0x00,0x48]
# DIS: 48000ca6{{[ \t]+}}ttsfpswap l2, l3, 9
