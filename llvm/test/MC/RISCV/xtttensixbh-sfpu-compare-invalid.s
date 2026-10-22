# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

ttsfpgt l0, l1, 0
# CHECK: error: SFPGT mode must be one of {1}
ttsfple l0, l1, 2
# CHECK: error: SFPLE mode must be one of {1}
ttsfpgt l0, l1, 8
# CHECK: error: SFPGT mode must be one of {1}
ttsfple l0, l1, 9
# CHECK: error: SFPLE mode must be one of {1}
ttsfpgt c8, l1, 1
# CHECK: error: invalid operand for instruction
ttsfple c12, l0, 1
# CHECK: error: invalid operand for instruction
ttsfpgt l0, a0, 1
# CHECK: error: invalid operand for instruction
