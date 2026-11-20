# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=VALID
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

#--- valid.s
ttatgetm 0
ttatgetm 2
ttatrelm 3
ttatrelm 4

# VALID: ttatgetm 0{{ *}}# encoding: [0x02,0x00,0x00,0x80]
# VALID: ttatgetm 2{{ *}}# encoding: [0x0a,0x00,0x00,0x80]
# VALID: ttatrelm 3{{ *}}# encoding: [0x0e,0x00,0x00,0x84]
# VALID: ttatrelm 4{{ *}}# encoding: [0x12,0x00,0x00,0x84]

#--- invalid.s
ttatgetm 1
ttatrelm 5

# INVALID-COUNT-2: error: Blackhole mutex index must be 0, 2, 3, or 4
