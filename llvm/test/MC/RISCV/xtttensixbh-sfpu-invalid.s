# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

ttsfploadi l0, 0, 3
# CHECK: error: SFPLOADI mode must be one of {0, 1, 2, 4, 8, 10}
ttsfploadi l0, 0, 15
# CHECK: error: SFPLOADI mode must be one of {0, 1, 2, 4, 8, 10}
ttsfploadi l0, 65536, 0
# CHECK: error: immediate must be an integer in the range [0, 65535]
ttsfploadi l0, -1, 0
# CHECK: error: immediate must be an integer in the range [0, 65535]
ttsfpencc 0, 3
# CHECK: error: SFPENCC mode must be one of {0, 1, 2, 8, 9, 10}
ttsfpencc 3, 15
# CHECK: error: SFPENCC mode must be one of {0, 1, 2, 8, 9, 10}
ttsfpencc 4, 0
# CHECK: error: immediate must be an integer in the range [0, 3]
ttsfpmad l0, l1, l2, l3, 4
# CHECK: error: immediate must be an integer in the range [0, 3]
ttsfpadd l0, l1, l2, 4
# CHECK: error: immediate must be an integer in the range [0, 3]
ttsfpmov.all c8, l0
# CHECK: error: invalid operand for instruction
ttsfploadi c15, 0, 0
# CHECK: error: invalid operand for instruction
ttsfpmov.all l0, a0
# CHECK: error: invalid operand for instruction
ttsfpiadd l0, ttstage
# CHECK: error: invalid operand for instruction
ttsfpsetcc.eq0 ttcc
# CHECK: error: invalid operand for instruction
ttsfppushc 0
# CHECK: error: unexpected extra operand for instruction
ttsfpload l0, 1024, 7, 3
# CHECK: error: immediate must be an integer in the range [0, 1023]
ttsfpstore l0, -1, 7, 3
# CHECK: error: immediate must be an integer in the range [0, 1023]
ttsfpload l0, 0, 8, 3
# CHECK: error: immediate must be an integer in the range [0, 7]
ttsfpstore l0, 0, 7, 1
# CHECK: error: SFPSTORE format must be one of {0, 2, 3, 4}
ttsfpload l0, 0, 7, 5
# CHECK: error: SFPLOAD format must be one of {0, 2, 3, 4}
