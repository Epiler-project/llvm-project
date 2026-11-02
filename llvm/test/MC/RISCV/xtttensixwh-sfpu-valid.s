# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=NOFEATURE

# Known bytes from official WormholeB0 SFPU diagrams: direct issue rotates
# the raw word left by two. AddrMod3 occupies bits15:14, unlike BH15:13.
# MAD low modifier bits are reserved on WH; FP32 ABS uses mode1.

# ENC: ttsfpencc 3, 10 # encoding: [0x2a,0xc0,0x00,0x28]
# DIS: {{.*}}ttsfpencc 3, 10
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpencc 3, 10

# ENC: ttsfpconfig.reset # encoding: [0xc6,0x03,0x00,0x44]
# DIS: {{.*}}ttsfpconfig.reset
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpconfig.reset

# ENC: ttsfploadi l0, 16384, 0 # encoding: [0x01,0x00,0x01,0xc4]
# DIS: {{.*}}ttsfploadi l0, 16384, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfploadi l0, 16384, 0

# ENC: ttsfpmov l1, l2 # encoding: [0x41,0x08,0x00,0xf0]
# DIS: {{.*}}ttsfpmov l1, l2
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpmov l1, l2

# ENC: ttsfpmov.neg l3, c8 # encoding: [0xc5,0x20,0x00,0xf0]
# DIS: {{.*}}ttsfpmov.neg l3, c8
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpmov.neg l3, c8

# ENC: ttsfpmov.all l7, c15 # encoding: [0xc9,0x3d,0x00,0xf0]
# DIS: {{.*}}ttsfpmov.all l7, c15
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpmov.all l7, c15

# ENC: ttsfpmad l3, l1, c10, l7, 0 # encoding: [0xc2,0x9c,0x06,0x10]
# DIS: {{.*}}ttsfpmad l3, l1, c10, l7, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpmad l3, l1, c10, l7, 0

# ENC: ttsfpmul l5, l2, c12, 0 # encoding: [0x42,0x25,0x0b,0x18]
# DIS: {{.*}}ttsfpmul l5, l2, c12, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpmul l5, l2, c12, 0

# ENC: ttsfpadd l5, l2, c12, 0 # encoding: [0x42,0xb1,0x28,0x14]
# DIS: {{.*}}ttsfpadd l5, l2, c12, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpadd l5, l2, c12, 0

# ENC: ttsfpnop # encoding: [0x02,0x00,0x00,0x3c]
# DIS: {{.*}}ttsfpnop
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpnop

# ENC: ttsfpabs l3, l7, 1 # encoding: [0xc5,0x1c,0x00,0xf4]
# DIS: {{.*}}ttsfpabs l3, l7, 1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpabs l3, l7, 1

# ENC: ttsfpxor l3, l7 # encoding: [0xc2,0x1c,0x00,0x34]
# DIS: {{.*}}ttsfpxor l3, l7
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpxor l3, l7

# ENC: ttsfpload l0, 0, 0, 3 # encoding: [0x01,0x00,0x0c,0xc0]
# DIS: {{.*}}ttsfpload l0, 0, 0, 3
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpload l0, 0, 0, 3

# ENC: ttsfpload l7, 1023, 3, 3 # encoding: [0xfd,0x0f,0xcf,0xc1]
# DIS: {{.*}}ttsfpload l7, 1023, 3, 3
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpload l7, 1023, 3, 3

# ENC: ttsfpstore l0, 0, 0, 3 # encoding: [0x01,0x00,0x0c,0xc8]
# DIS: {{.*}}ttsfpstore l0, 0, 0, 3
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpstore l0, 0, 0, 3

# ENC: ttsfpstore l7, 1023, 3, 3 # encoding: [0xfd,0x0f,0xcf,0xc9]
# DIS: {{.*}}ttsfpstore l7, 1023, 3, 3
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpstore l7, 1023, 3, 3
