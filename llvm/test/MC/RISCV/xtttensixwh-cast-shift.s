# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=NOFEATURE

# Literal encodings from official WH SFPCAST/SFPSHFT bit diagrams, rotated left
# by two for direct issue. CAST only modes0/1; SHFT uses a signed12 immediate.
# ENC: ttsfpcast l0, l1, 0 # encoding: [0x02,0x04,0x00,0x40]
# DIS: ttsfpcast l0, l1, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpcast l0, l1, 0
# ENC: ttsfpcast l7, c15, 1 # encoding: [0xc6,0x3d,0x00,0x40]
# DIS: ttsfpcast l7, c15, 1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpcast l7, c15, 1
# ENC: ttsfpshft l0, l1, 0, 0 # encoding: [0x01,0x04,0x00,0xe8]
# DIS: ttsfpshft l0, l1, 0, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpshft l0, l1, 0, 0
# ENC: ttsfpshft l7, l7, -2048, 1 # encoding: [0xc5,0x1d,0x00,0xea]
# DIS: ttsfpshft l7, l7, -2048, 1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpshft l7, l7, -2048, 1
# ENC: ttsfpshft l2, c8, 2047, 1 # encoding: [0x85,0xe0,0xff,0xe9]
# DIS: ttsfpshft l2, c8, 2047, 1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpshft l2, c8, 2047, 1
# ENC: ttsfpshft l3, c10, -1, 1 # encoding: [0xc5,0xe8,0xff,0xeb]
# DIS: ttsfpshft l3, c10, -1, 1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpshft l3, c10, -1, 1
