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
# CHECK: ttrmwcib0 2, 0, 3
# CHECK: ttrmwcib0 2, 1, 3
# CHECK: ttrmwcib0 0, 20, 31
# CHECK: ttrmwcib0 0, 160, 224
# CHECK: ttrmwcib1 0, 2, 3
# CHECK: ttrmwcib1 0, 0, 64
# CHECK: ttrmwcib0 1, 0, 2
# CHECK: ttrmwcib3 1, 0, 30
# CHECK: ttrmwcib3 1, 10, 30
# CHECK: ttrmwcib3 1, 16, 30
# CHECK: ttrmwcib3 1, 18, 30
# INVALID-COUNT-10: error: Wormhole RMWCIB0 does not match an admitted logical form
# INVALID-COUNT-3: error: Wormhole RMWCIB1 does not match an admitted logical form
# INVALID-COUNT-2: error: Wormhole RMWCIB3 does not match an admitted logical form

#--- valid.s
ttrmwcib0 2, 4, 60
ttrmwcib0 2, 0, 60
ttrmwcib0 6, 0, 255

ttrmwcib3 1, 32, 32
ttrmwcib3 1, 64, 64

ttrmwcib0 0, 16, 31
ttrmwcib0 0, 21, 31
ttrmwcib0 2, 0, 3
ttrmwcib0 2, 1, 3
ttrmwcib0 0, 20, 31
ttrmwcib0 0, 160, 224
ttrmwcib1 0, 2, 3

ttrmwcib1 0, 0, 64
ttrmwcib0 1, 0, 2
ttrmwcib3 1, 0, 30
ttrmwcib3 1, 10, 30
ttrmwcib3 1, 16, 30
ttrmwcib3 1, 18, 30

#--- invalid.s
# A threshold mode, the adjacent ALU control bits, and unrelated words stay out.
ttrmwcib0 2, 8, 60
ttrmwcib0 2, 4, 63
ttrmwcib0 3, 4, 60
# SrcA format override must keep the SrcB bits and use an admitted format.
ttrmwcib0 0, 16, 63
ttrmwcib0 0, 17, 31
# Pack format setup must preserve neighboring FPU rounding, SrcB and Math
# precision bits. An unsupported Dst format also fails its exact relation.
ttrmwcib0 1, 0, 3
ttrmwcib0 2, 2, 3
ttrmwcib0 2, 1, 7
ttrmwcib0 0, 160, 240
ttrmwcib0 0, 128, 224
ttrmwcib1 0, 0, 66
ttrmwcib1 0, 2, 7
ttrmwcib1 0, 1, 3
ttrmwcib3 1, 10, 62
ttrmwcib3 1, 4, 30
