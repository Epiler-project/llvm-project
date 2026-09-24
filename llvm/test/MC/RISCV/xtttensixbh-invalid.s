# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

ttsetc16 0, 68
# CHECK: error: SETC16 field setc16_reg must be in [0, 67]
ttsetc16 0, -1
# CHECK: error: invalid operand for instruction
ttsetc16 65536, 1
# CHECK: error: SETC16 field setc16_value must be in [0, 65535]
