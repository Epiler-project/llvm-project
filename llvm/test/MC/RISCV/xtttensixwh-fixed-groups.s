# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# Official raw0x94: copy4 modes0/1; rotate4 mode2 with VC at[11:8].
# Official raw0x8c: transpose at VD0. Direct issue rotates the raw word left2.
ttsfpshft2.copy4 0
# ENC: ttsfpshft2.copy4 0 # encoding: [0x02,0x00,0x00,0x50]
# DIS: 50000002{{[ \t]+}}ttsfpshft2.copy4 0
ttsfpshft2.copy4 1
# ENC: ttsfpshft2.copy4 1 # encoding: [0x06,0x00,0x00,0x50]
# DIS: 50000006{{[ \t]+}}ttsfpshft2.copy4 1
ttsfpshft2.rotate4 l0
# ENC: ttsfpshft2.rotate4 l0 # encoding: [0x0a,0x00,0x00,0x50]
# DIS: 5000000a{{[ \t]+}}ttsfpshft2.rotate4 l0
ttsfpshft2.rotate4 c15
# ENC: ttsfpshft2.rotate4 c15 # encoding: [0x0a,0x3c,0x00,0x50]
# DIS: 50003c0a{{[ \t]+}}ttsfpshft2.rotate4 c15
ttsfptransp
# ENC: ttsfptransp # encoding: [0x02,0x00,0x00,0x30]
# DIS: 30000002{{[ \t]+}}ttsfptransp
