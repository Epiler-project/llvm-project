# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %s 2>&1 | FileCheck %s

# WH rejects BH negate forms, even though their bit positions are shared.
# CHECK: error: Wormhole SFPADDI operand 3 must be in [0, 0]
ttsfpaddi l0, 16256, 2
# CHECK: error: Wormhole SFPMULI operand 3 must be in [0, 0]
ttsfpmuli l0, 16384, 2
# CHECK: error: instruction requires the following:
ttsfpmul24 l0, l0, c10, 0
