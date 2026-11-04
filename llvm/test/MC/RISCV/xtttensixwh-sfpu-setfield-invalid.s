# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

# CHECK: error: Tensix field immediate must be in [0, 255]
ttsfpsetexp l2, c9, 256, 1

# CHECK: error: Tensix field immediate must be in [0, 0]
ttsfpsetexp l2, c9, 1, 0

# CHECK: error: SFPSETEXP mode must be one of
ttsfpsetexp l2, c9, 0, 4

# CHECK: error: Tensix field immediate must be in [0, 4095]
ttsfpsetman l2, c9, 4096, 1

# CHECK: error: Tensix field immediate must be in [0, 0]
ttsfpsetman l2, c9, 1, 0

# CHECK: error: SFPSETMAN mode must be one of
ttsfpsetman l2, c9, 0, 2

# CHECK: error: Tensix field immediate must be in [0, 1]
ttsfpsetsgn l2, c9, 2, 1

# CHECK: error: Tensix field immediate must be in [0, 0]
ttsfpsetsgn l2, c9, 1, 0

# CHECK: error: SFPSETSGN mode must be one of
ttsfpsetsgn l2, c9, 0, 2
