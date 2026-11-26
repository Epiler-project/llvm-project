# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS

# ENC: ttsfpdivp2 l2, c15, 0, 0 # encoding: [0x81,0x3c,0x00,0xd8]
# DIS: {{.*}}ttsfpdivp2 l2, c15, 0, 0
ttsfpdivp2 l2, c15, 0, 0

# ENC: ttsfpdivp2 l7, c9, 255, 1 # encoding: [0xc5,0xe5,0x3f,0xd8]
# DIS: {{.*}}ttsfpdivp2 l7, c9, 255, 1
ttsfpdivp2 l7, c9, 255, 1

# ENC: ttsfpaddi l2, 16256, 0 # encoding: [0x81,0x00,0xfe,0xd4]
# DIS: {{.*}}ttsfpaddi l2, 16256, 0
ttsfpaddi l2, 16256, 0

# ENC: ttsfpmuli l7, 65535, 0 # encoding: [0xc1,0xfd,0xff,0xd3]
# DIS: {{.*}}ttsfpmuli l7, 65535, 0
ttsfpmuli l7, 65535, 0
