# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# Raw words checked against Apache SDK ckernel_ops.h and independent ISA docs.
# ENC: ttsfpmul l3, l1, l2, 2 # encoding: [0xca,0xa4,0x04,0x18]
# DIS: ttsfpmul l3, l1, l2, 2
ttsfpmul l3, l1, l2, 2
# ENC: ttsfpcompc # encoding: [0x02,0x00,0x00,0x2c]
# DIS: ttsfpcompc
ttsfpcompc
# ENC: ttsfpconfig.reset # encoding: [0xc6,0x03,0x00,0x44]
# DIS: ttsfpconfig.reset
ttsfpconfig.reset
# ENC: ttsfpconfig.c11 0, 0 # encoding: [0xc2,0x02,0x00,0x44]
# DIS: ttsfpconfig.c11 0, 0
ttsfpconfig.c11 0, 0
# ENC: ttsfpconfig.c11 21845, 8 # encoding: [0xe2,0x56,0x55,0x45]
# DIS: ttsfpconfig.c11 21845, 8
ttsfpconfig.c11 21845, 8
# Check zero and individual even lane bits as well as the full mask above.
# ENC: ttsfpconfig.c12 1, 8 # encoding: [0x22,0x07,0x00,0x44]
# DIS: ttsfpconfig.c12 1, 8
ttsfpconfig.c12 1, 8
# ENC: ttsfpconfig.c13 16384, 8 # encoding: [0x62,0x03,0x00,0x45]
# DIS: ttsfpconfig.c13 16384, 8
ttsfpconfig.c13 16384, 8
# ENC: ttsfpconfig.c14 4, 8 # encoding: [0xa2,0x13,0x00,0x44]
# DIS: ttsfpconfig.c14 4, 8
ttsfpconfig.c14 4, 8
# ENC: ttsfpconfig.c14 0, 8 # encoding: [0xa2,0x03,0x00,0x44]
# DIS: ttsfpconfig.c14 0, 8
ttsfpconfig.c14 0, 8
