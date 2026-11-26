# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# Official SFPSHFT2: raw0x94 | VC[11:8] | VD[7:4] | mode3, rotated left2.
ttsfpshft2 l0, l1, 3
# ENC: ttsfpshft2 l0, l1, 3 # encoding: [0x0e,0x04,0x00,0x50]
# DIS: 5000040e{{[ \t]+}}ttsfpshft2 l0, l1, 3
ttsfpshft2 l7, c15, 3
# ENC: ttsfpshft2 l7, c15, 3 # encoding: [0xce,0x3d,0x00,0x50]
# DIS: 50003dce{{[ \t]+}}ttsfpshft2 l7, c15, 3
