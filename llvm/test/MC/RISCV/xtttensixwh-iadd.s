# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/good.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/good.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %t/good.s 2>&1 | FileCheck %s --check-prefix=NOFEATURE
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/bad.s 2>&1 | FileCheck %s --check-prefix=BAD

# SFPIADD has the same published functional model and encoding on WH/BH.
# Raw opcode 0x79 uses Imm12[23:12], VC[11:8], VD[7:4], Mod1[3:0];
# direct issue rotates that word left by two. These literal bytes cover all
# admitted CC/add/subtract forms and both signed immediate endpoints.

#--- good.s
# ENC: ttsfpiadd.lt0 l0, l1 # encoding: [0x01,0x04,0x00,0xe4]
# DIS: ttsfpiadd.lt0 l0, l1
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttsfpiadd.lt0 l0, l1
# ENC: ttsfpiadd.sub.lt0 l2, l3 # encoding: [0x89,0x0c,0x00,0xe4]
# DIS: ttsfpiadd.sub.lt0 l2, l3
ttsfpiadd.sub.lt0 l2, l3
# ENC: ttsfpiadd l4, c15 # encoding: [0x11,0x3d,0x00,0xe4]
# DIS: ttsfpiadd l4, c15
ttsfpiadd l4, c15
# ENC: ttsfpiadd.sub l5, c8 # encoding: [0x59,0x21,0x00,0xe4]
# DIS: ttsfpiadd.sub l5, c8
ttsfpiadd.sub l5, c8
# ENC: ttsfpiadd.ge0 l6, l7 # encoding: [0xa1,0x1d,0x00,0xe4]
# DIS: ttsfpiadd.ge0 l6, l7
ttsfpiadd.ge0 l6, l7
# ENC: ttsfpiadd.sub.ge0 l7, l0 # encoding: [0xe9,0x01,0x00,0xe4]
# DIS: ttsfpiadd.sub.ge0 l7, l0
ttsfpiadd.sub.ge0 l7, l0
# ENC: ttsfpiadd.i l0, l0, -2048, 1 # encoding: [0x05,0x00,0x00,0xe6]
# DIS: ttsfpiadd.i l0, l0, -2048, 1
ttsfpiadd.i l0, l0, -2048, 1
# ENC: ttsfpiadd.i l7, c8, 2047, 5 # encoding: [0xd5,0xe1,0xff,0xe5]
# DIS: ttsfpiadd.i l7, c8, 2047, 5
ttsfpiadd.i l7, c8, 2047, 5
# ENC: ttsfpiadd.i l3, l3, 0, 9 # encoding: [0xe5,0x0c,0x00,0xe4]
# DIS: ttsfpiadd.i l3, l3, 0, 9
ttsfpiadd.i l3, l3, 0, 9

#--- bad.s
ttsfpiadd.i l0, l1, -2049, 5
# BAD: error: immediate must be an integer in the range [-2048, 2047]
ttsfpiadd.i l0, l1, 2048, 5
# BAD: error: immediate must be an integer in the range [-2048, 2047]
ttsfpiadd.i l0, l1, 0, 0
# BAD: error: SFPIADDI mode must be one of {1, 5, 9}
ttsfpiadd.i l0, l1, 0, 13
# BAD: error: SFPIADDI mode must be one of {1, 5, 9}
ttsfpiadd c9, l1
# BAD: error: invalid operand for instruction
ttsfpiadd l0, ttstage
# BAD: error: invalid operand for instruction
