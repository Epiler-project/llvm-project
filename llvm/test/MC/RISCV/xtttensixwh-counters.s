# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/bad.s 2>&1 | FileCheck %s --check-prefix=ERR

# Independently assembled from Wormhole LLK assembly.yaml and ckernel_ops.h.
# CHECK: ttsetrwc 13, 3, 5, 9, 6, 2{{ *}}# encoding: [0x34,0x53,0x69,0xde]
# CHECK: ttsetibrwc 1, 4095, 1{{ *}}# encoding: [0x04,0xff,0x1f,0xe4]
# DIS: ttsetrwc 13, 3, 5, 9, 6, 2
# DIS: ttsetibrwc 1, 4095, 1
# ERR: error: Wormhole SETRWC field bitmask
# ERR: error: Wormhole SETIBRWC field set_inc_ctrl
# ERR: error: Wormhole SETIBRWC field rwc_cr

#--- valid.s
ttsetrwc 13, 3, 5, 9, 6, 2
ttsetibrwc 1, 4095, 1

#--- bad.s
ttsetrwc 16, 0, 0, 0, 0, 0
ttsetibrwc 2, 0, 0
ttsetibrwc 0, 0, 2
