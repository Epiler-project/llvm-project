# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# WH raw opcode0x95 has VD[7:4], Mod1[3:0], and no Mod1Mirror.
# Direct issue is raw word rotated left2.

ttsfplut l0, 0
# ENC: ttsfplut l0, 0 # encoding: [0x01,0x00,0x00,0xcc]
# DIS: cc000001{{[[:space:]]+}}ttsfplut l0, 0

ttsfplut l7, 4
# ENC: ttsfplut l7, 4 # encoding: [0x01,0x00,0xd0,0xcd]
# DIS: cdd00001{{[[:space:]]+}}ttsfplut l7, 4

ttsfplutfp32.6r l0, 0
# ENC: ttsfplutfp32.6r l0, 0 # encoding: [0x02,0x00,0x00,0x54]
# DIS: 54000002{{[[:space:]]+}}ttsfplutfp32.6r l0, 0

ttsfplutfp32.6r l7, 0
# ENC: ttsfplutfp32.6r l7, 0 # encoding: [0xc2,0x01,0x00,0x54]
# DIS: 540001c2{{[[:space:]]+}}ttsfplutfp32.6r l7, 0

ttsfplutfp32.6r l3, 2
# ENC: ttsfplutfp32.6r l3, 2 # encoding: [0xca,0x00,0x00,0x54]
# DIS: 540000ca{{[[:space:]]+}}ttsfplutfp32.6r l3, 2

ttsfplutfp32.6r l4, 3
# ENC: ttsfplutfp32.6r l4, 3 # encoding: [0x0e,0x01,0x00,0x54]
# DIS: 5400010e{{[[:space:]]+}}ttsfplutfp32.6r l4, 3

ttsfplutfp32.6r l5, 4
# ENC: ttsfplutfp32.6r l5, 4 # encoding: [0x52,0x01,0x00,0x54]
# DIS: 54000152{{[[:space:]]+}}ttsfplutfp32.6r l5, 4

ttsfplutfp32.6r l6, 6
# ENC: ttsfplutfp32.6r l6, 6 # encoding: [0x9a,0x01,0x00,0x54]
# DIS: 5400019a{{[[:space:]]+}}ttsfplutfp32.6r l6, 6

ttsfplutfp32.6r l7, 7
# ENC: ttsfplutfp32.6r l7, 7 # encoding: [0xde,0x01,0x00,0x54]
# DIS: 540001de{{[[:space:]]+}}ttsfplutfp32.6r l7, 7

ttsfplutfp32.3r 0
# ENC: ttsfplutfp32.3r 0 # encoding: [0x2a,0x00,0x00,0x54]
# DIS: 5400002a{{[[:space:]]+}}ttsfplutfp32.3r 0

ttsfplutfp32.3r 1
# ENC: ttsfplutfp32.3r 1 # encoding: [0x3a,0x00,0x00,0x54]
# DIS: 5400003a{{[[:space:]]+}}ttsfplutfp32.3r 1
