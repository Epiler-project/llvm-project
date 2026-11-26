# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s

# Literal encodings from opcode0x95, VD[7:4], Mod1[3:0], mirror[19:16],
# followed by the documented native 32-bit rotate-left-two encoding.

# CHECK: ttsfplutfp32.6r l0, 0 # encoding: [0x02,0x00,0x00,0x54]
ttsfplutfp32.6r l0, 0

# CHECK: ttsfplutfp32.6r l7, 0 # encoding: [0xc2,0x01,0x00,0x54]
ttsfplutfp32.6r l7, 0

# CHECK: ttsfplutfp32.6r l3, 2 # encoding: [0xca,0x00,0x08,0x54]
ttsfplutfp32.6r l3, 2

# CHECK: ttsfplutfp32.6r l4, 3 # encoding: [0x0e,0x01,0x0c,0x54]
ttsfplutfp32.6r l4, 3

# CHECK: ttsfplutfp32.6r l5, 4 # encoding: [0x52,0x01,0x10,0x54]
ttsfplutfp32.6r l5, 4

# CHECK: ttsfplutfp32.6r l6, 6 # encoding: [0x9a,0x01,0x18,0x54]
ttsfplutfp32.6r l6, 6

# CHECK: ttsfplutfp32.6r l7, 7 # encoding: [0xde,0x01,0x1c,0x54]
ttsfplutfp32.6r l7, 7

# CHECK: ttsfplutfp32.3r 0 # encoding: [0x2a,0x00,0x28,0x54]
ttsfplutfp32.3r 0

# CHECK: ttsfplutfp32.3r 1 # encoding: [0x3a,0x00,0x38,0x54]
ttsfplutfp32.3r 1
