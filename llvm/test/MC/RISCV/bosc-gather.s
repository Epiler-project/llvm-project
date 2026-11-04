# RUN: llvm-mc -triple=riscv64 -mattr=+v -show-encoding %s | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 | FileCheck %s --check-prefix=NOV
# CHECK: vrgather.vx v2, v1, a0 # encoding: [0x57,0x41,0x15,0x32]
# NOV: instruction requires the following: 'V'
vrgather.vx v2, v1, a0
