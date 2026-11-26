# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

#--- valid.s
ttreg2flop 62, 5, 0, 0, 0, 1
ttreg2flop 63, 99, 0, 0, 0, 1
# CHECK: ttreg2flop 62, 5, 0, 0, 0, 1{{ *}}# encoding: [0xf9,0x05,0x00,0x21]
# CHECK: ttreg2flop 63, 99, 0, 0, 0, 1{{ *}}# encoding: [0xfd,0x63,0x00,0x21]

#--- invalid.s
ttreg2flop 62, 100, 0, 0, 0, 1
ttreg2flop 62, 5, 0, 0, 2, 1
ttreg2flop 62, 5, 0, 0, 0, 0
ttreg2flop 62, 5, 0, 0, 0, 2
# INVALID: error: {{.*}}flopindex{{.*}}99
# INVALID: error: {{.*}}targetsel{{.*}}0
# INVALID: error:
# INVALID: error:
