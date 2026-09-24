# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=NOFEATURE

# Raw SFPU instruction words are rotated left by two for direct issue. These
# byte literals also distinguish the tied destination from arithmetic sources.
# Ordinary RV32 instructions remain four bytes in the same instruction stream.
# DIS: 0: {{.*}}addi a0, a0, 1
addi a0, a0, 1

# ENC: ttsfploadi l0, 0, 0 # encoding: [0x01,0x00,0x00,0xc4]
# DIS: 4: {{.*}}ttsfploadi l0, 0, 0
# NOFEATURE: instruction requires the following: 'XTTTensixBH'
ttsfploadi l0, 0, 0
# ENC: ttsfploadi l7, 65535, 10 # encoding: [0xfd,0xff,0xeb,0xc5]
# DIS: 8: {{.*}}ttsfploadi l7, 65535, 10
ttsfploadi l7, 65535, 10
# ENC: ttsfpmov l1, l2 # encoding: [0x41,0x08,0x00,0xf0]
# DIS: c: {{.*}}ttsfpmov l1, l2
ttsfpmov l1, l2
# ENC: ttsfpmov.neg l3, c8 # encoding: [0xc5,0x20,0x00,0xf0]
# DIS: 10: {{.*}}ttsfpmov.neg l3, c8
ttsfpmov.neg l3, c8
# ENC: ttsfpmov.all l7, c15 # encoding: [0xc9,0x3d,0x00,0xf0]
# DIS: 14: {{.*}}ttsfpmov.all l7, c15
ttsfpmov.all l7, c15
# ENC: ttsfpmad l3, l1, c10, l7, 3 # encoding: [0xce,0x9c,0x06,0x10]
# DIS: 18: {{.*}}ttsfpmad l3, l1, c10, l7, 3
ttsfpmad l3, l1, c10, l7, 3
# ENC: ttsfpadd l5, l2, c12, 2 # encoding: [0x4a,0xb1,0x28,0x14]
# DIS: 1c: {{.*}}ttsfpadd l5, l2, c12, 2
ttsfpadd l5, l2, c12, 2
# ENC: ttsfpiadd.lt0 l0, c8 # encoding: [0x01,0x20,0x00,0xe4]
# DIS: 20: {{.*}}ttsfpiadd.lt0 l0, c8
ttsfpiadd.lt0 l0, c8
# ENC: ttsfpiadd.sub.lt0 l1, l2 # encoding: [0x49,0x08,0x00,0xe4]
# DIS: 24: {{.*}}ttsfpiadd.sub.lt0 l1, l2
ttsfpiadd.sub.lt0 l1, l2
# ENC: ttsfpiadd l2, l3 # encoding: [0x91,0x0c,0x00,0xe4]
# DIS: 28: {{.*}}ttsfpiadd l2, l3
ttsfpiadd l2, l3
# ENC: ttsfpiadd.sub l3, l4 # encoding: [0xd9,0x10,0x00,0xe4]
# DIS: 2c: {{.*}}ttsfpiadd.sub l3, l4
ttsfpiadd.sub l3, l4
# ENC: ttsfpiadd.ge0 l4, l5 # encoding: [0x21,0x15,0x00,0xe4]
# DIS: 30: {{.*}}ttsfpiadd.ge0 l4, l5
ttsfpiadd.ge0 l4, l5
# ENC: ttsfpiadd.sub.ge0 l5, l6 # encoding: [0x69,0x19,0x00,0xe4]
# DIS: 34: {{.*}}ttsfpiadd.sub.ge0 l5, l6
ttsfpiadd.sub.ge0 l5, l6
# ENC: ttsfppushc # encoding: [0x02,0x00,0x00,0x1c]
# DIS: 38: {{.*}}ttsfppushc
ttsfppushc
# ENC: ttsfppopc # encoding: [0x02,0x00,0x00,0x20]
# DIS: 3c: {{.*}}ttsfppopc
ttsfppopc
# ENC: ttsfpnop # encoding: [0x02,0x00,0x00,0x3c]
# DIS: 40: {{.*}}ttsfpnop
ttsfpnop
# ENC: ttsfpencc 3, 10 # encoding: [0x2a,0xc0,0x00,0x28]
# DIS: 44: {{.*}}ttsfpencc 3, 10
ttsfpencc 3, 10
# ENC: ttsfpsetcc.lt0 l0 # encoding: [0x01,0x00,0x00,0xec]
# DIS: 48: {{.*}}ttsfpsetcc.lt0 l0
ttsfpsetcc.lt0 l0
# ENC: ttsfpsetcc.ne0 l7 # encoding: [0x09,0x1c,0x00,0xec]
# DIS: 4c: {{.*}}ttsfpsetcc.ne0 l7
ttsfpsetcc.ne0 l7
# ENC: ttsfpsetcc.ge0 c8 # encoding: [0x11,0x20,0x00,0xec]
# DIS: 50: {{.*}}ttsfpsetcc.ge0 c8
ttsfpsetcc.ge0 c8
# ENC: ttsfpsetcc.eq0 c15 # encoding: [0x19,0x3c,0x00,0xec]
# DIS: 54: {{.*}}ttsfpsetcc.eq0 c15
ttsfpsetcc.eq0 c15
# ENC: ttsfpload l0, 0, 0, 2 # encoding: [0x01,0x00,0x08,0xc0]
# DIS: 58: {{.*}}ttsfpload l0, 0, 0, 2
ttsfpload l0, 0, 0, 2
# ENC: ttsfpload l7, 1023, 7, 3 # encoding: [0xfd,0x8f,0xcf,0xc1]
# DIS: 5c: {{.*}}ttsfpload l7, 1023, 7, 3
ttsfpload l7, 1023, 7, 3
# ENC: ttsfpload l3, 513, 2, 4 # encoding: [0x05,0x08,0xd1,0xc0]
# DIS: 60: {{.*}}ttsfpload l3, 513, 2, 4
ttsfpload l3, 513, 2, 4
# ENC: ttsfpload l0, 0, 0, 0 # encoding: [0x01,0x00,0x00,0xc0]
# DIS: 64: {{.*}}ttsfpload l0, 0, 0, 0
ttsfpload l0, 0, 0, 0
# ENC: ttsfpstore l0, 0, 0, 2 # encoding: [0x01,0x00,0x08,0xc8]
# DIS: 68: {{.*}}ttsfpstore l0, 0, 0, 2
ttsfpstore l0, 0, 0, 2
# ENC: ttsfpstore l7, 1023, 7, 3 # encoding: [0xfd,0x8f,0xcf,0xc9]
# DIS: 6c: {{.*}}ttsfpstore l7, 1023, 7, 3
ttsfpstore l7, 1023, 7, 3
# ENC: ttsfpstore l3, 513, 2, 4 # encoding: [0x05,0x08,0xd1,0xc8]
# DIS: 70: {{.*}}ttsfpstore l3, 513, 2, 4
ttsfpstore l3, 513, 2, 4
# ENC: ttsfpstore l0, 0, 0, 0 # encoding: [0x01,0x00,0x00,0xc8]
# DIS: 74: {{.*}}ttsfpstore l0, 0, 0, 0
ttsfpstore l0, 0, 0, 0
# DIS: 78: {{.*}}ret
ret
