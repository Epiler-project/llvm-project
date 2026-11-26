# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s

# Literal raw words above the rotate boundary are independently derived
# from the published Blackhole instruction diagrams, not a schema importer.

# CHECK: ttsfparecip l2, l3, 1 # encoding: [0x86,0x8c,0x00,0x64]
ttsfparecip l2, l3, 1
# CHECK: ttsfpexexp l7, c15, 11 # encoding: [0xed,0x3d,0x00,0xdc]
ttsfpexexp l7, c15, 11
# CHECK: ttsfpexman l7, c15, 1 # encoding: [0xc5,0x3d,0x00,0xe0]
ttsfpexman l7, c15, 1
# CHECK: ttsfpabs l7, c15, 1 # encoding: [0xc5,0x3d,0x00,0xf4]
ttsfpabs l7, c15, 1
# CHECK: ttsfplz l7, c15, 14 # encoding: [0xfa,0x3d,0x00,0x04]
ttsfplz l7, c15, 14
# CHECK: ttsfpcast l7, c15, 3 # encoding: [0xce,0x3d,0x00,0x40]
ttsfpcast l7, c15, 3
# CHECK: ttsfpsetexp l7, c15, 255, 1 # encoding: [0xc6,0xfd,0x3f,0x08]
ttsfpsetexp l7, c15, 255, 1
# CHECK: ttsfpsetman l7, c15, 4095, 1 # encoding: [0xc6,0xfd,0xff,0x0f]
ttsfpsetman l7, c15, 4095, 1
# CHECK: ttsfpsetsgn l7, c15, 1, 1 # encoding: [0xc6,0x7d,0x00,0x24]
ttsfpsetsgn l7, c15, 1, 1
# CHECK: ttsfpand l7, c15 # encoding: [0xc1,0x3d,0x00,0xf8]
ttsfpand l7, c15
# CHECK: ttsfpor l7, c15 # encoding: [0xc1,0x3d,0x00,0xfc]
ttsfpor l7, c15
# CHECK: ttsfpxor l7, c15 # encoding: [0xc2,0x3d,0x00,0x34]
ttsfpxor l7, c15
# CHECK: ttsfpnot l7, c15 # encoding: [0xc2,0x3d,0x00,0x00]
ttsfpnot l7, c15
# CHECK: ttsfpiadd.i l7, c8, -2048, 5 # encoding: [0xd5,0x21,0x00,0xe6]
ttsfpiadd.i l7, c8, -2048, 5
# CHECK: ttsfpshft l7, c8, 2047, 7 # encoding: [0xdd,0xe1,0xff,0xe9]
ttsfpshft l7, c8, 2047, 7
# CHECK: ttsfpshft2 l7, c15, 3 # encoding: [0xce,0x3d,0x00,0x50]
ttsfpshft2 l7, c15, 3
# CHECK: ttsfpstochrnd.i l2, l3, 31, 13, 2 # encoding: [0xb6,0xcc,0x7c,0x39]
ttsfpstochrnd.i l2, l3, 31, 13, 2
# CHECK: ttsfpstochrnd.v l2, l3, l4, 5, 2 # encoding: [0x96,0x0c,0x01,0x39]
ttsfpstochrnd.v l2, l3, l4, 5, 2
# CHECK: ttsfpswap l2, l3, 9 # encoding: [0xa6,0x0c,0x00,0x48]
ttsfpswap l2, l3, 9
# CHECK: ttsfplut l7, 4 # encoding: [0x01,0x00,0xd0,0xcd]
ttsfplut l7, 4
# CHECK: ttsfptransp # encoding: [0x02,0x00,0x00,0x30]
ttsfptransp
