# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s

ttsfplutfp32.3r 2
# CHECK: error: immediate must be an integer in the range [0, 1]
ttsfplutfp32.3r -1
# CHECK: error: immediate must be an integer in the range [0, 1]
ttsfplutfp32.6r l0, 1
# CHECK: error: {{.*}}mode must be one of {0, 2, 3, 4, 6, 7}
ttsfplutfp32.6r l7, 5
# CHECK: error: {{.*}}mode must be one of {0, 2, 3, 4, 6, 7}
ttsfplutfp32.6r l7, 10
# CHECK: error: immediate must be an integer in the range [0, 7]
ttsfplutfp32.6r c8, 0
# CHECK: error: invalid operand for instruction
