# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %s 2>&1 | FileCheck %s

ttsfpmad l0, l1, l2, l3, 1
# CHECK: error: Wormhole SFPMAD operand 5 must be in [0, 0]
ttsfpmul l0, l1, l2, 2
# CHECK: error: Wormhole SFPMUL operand 4 must be in [0, 0]
ttsfpadd l0, l1, l2, 3
# CHECK: error: Wormhole SFPADD operand 4 must be in [0, 0]
ttsfpload l0, 0, 4, 3
# CHECK: error: Wormhole SFPLOAD operand 3 must be in [0, 3]
ttsfpstore l0, 0, 4, 3
# CHECK: error: Wormhole SFPSTORE operand 2 must be in [0, 3]
ttsfpabs l0, l1, 2
# CHECK: error: SFPABS mode must be one of {0, 1}
ttsfparecip l0, l1, 0
# CHECK: error: instruction requires the following: 'XTTTensixBH'
