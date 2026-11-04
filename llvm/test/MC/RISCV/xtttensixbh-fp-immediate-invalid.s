# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

# Direct immediate forms reject reserved/indirect mode bits at the MC owner.
# CHECK: error: SFPADDI mode must be one of {0, 2}
ttsfpaddi l0, 16256, 1
# CHECK: error: SFPADDI mode must be one of {0, 2}
ttsfpaddi l0, 16256, 8
# CHECK: error: SFPMULI mode must be one of {0, 2}
ttsfpmuli l0, 16384, 4
# CHECK: error: invalid operand for instruction
ttsfpaddi c9, 16256, 0
# CHECK: error: invalid operand for instruction
ttsfpmul24 c9, l0, c10, 0
