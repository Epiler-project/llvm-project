# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

# CHECK: ttrmwcib0 2, 4, 60{{ *}}# encoding: [0x0a,0x10,0xf0,0xcc]
# CHECK: ttrmwcib0 2, 0, 60{{ *}}# encoding: [0x0a,0x00,0xf0,0xcc]
# CHECK: ttrmwcib0 6, 0, 255{{ *}}# encoding: [0x1a,0x00,0xfc,0xcf]
# CHECK: ttrmwcib3 1, 32, 32
# CHECK: ttrmwcib3 1, 64, 64
# CHECK: ttrmwcib0 0, 16, 31
# CHECK: ttrmwcib0 0, 21, 31
# INVALID-COUNT-5: error: Wormhole RMWCIB0 does not match an admitted logical form

#--- valid.s
ttrmwcib0 2, 4, 60
ttrmwcib0 2, 0, 60
ttrmwcib0 6, 0, 255

ttrmwcib3 1, 32, 32
ttrmwcib3 1, 64, 64

ttrmwcib0 0, 16, 31
ttrmwcib0 0, 21, 31

#--- invalid.s
# A threshold mode, the adjacent ALU control bits, and unrelated words stay out.
ttrmwcib0 2, 8, 60
ttrmwcib0 2, 4, 63
ttrmwcib0 3, 4, 60
# SrcA format override must keep the SrcB bits and use an admitted format.
ttrmwcib0 0, 16, 63
ttrmwcib0 0, 17, 31
