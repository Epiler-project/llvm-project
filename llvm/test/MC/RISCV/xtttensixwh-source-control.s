# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

# Independently derived from pinned WH ckernel_ops.h and the functional models
# UNPACR_NOP_ZEROSRC, UNPACR_NOP_SETDVALID, CLEARDVALID, and SETRWC.
# The selected-bank wait-and-zero form is NoOp=0x11; it does not publish.
# Publication is a separate NoOp=0x7. Neither returns a Matrix-owned bank.
# WH: ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x45,0x00,0x00,0x0c]
# WH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0{{ *}}# encoding: [0x1d,0x00,0x00,0x0c]
# WH: ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 1{{ *}}# encoding: [0x45,0x00,0x00,0x0e]
# WH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1{{ *}}# encoding: [0x1d,0x00,0x00,0x0e]
# WH: ttsetrwc 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x04,0x00,0x00,0xdc]
# WH: ttcleardvalid 0, 1{{ *}}# encoding: [0x00,0x00,0x00,0xd9]
# WH: ttsetrwc 2, 0, 0, 0, 0, 0{{ *}}# encoding: [0x08,0x00,0x00,0xdc]
# WH: ttcleardvalid 0, 2{{ *}}# encoding: [0x00,0x00,0x00,0xda]
# BH: ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x85,0x00,0x00,0x0c]
# BH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0{{ *}}# encoding: [0x01,0x04,0x00,0x0c]
# BH: ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 1{{ *}}# encoding: [0x85,0x00,0x00,0x0e]
# BH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1{{ *}}# encoding: [0x01,0x04,0x00,0x0e]
# DIS: 0: {{.*}}ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 0
# DIS: 4: {{.*}}ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0
# DIS: 8: {{.*}}ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 1
# DIS: c: {{.*}}ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1
# DIS: 10: {{.*}}ttsetrwc 1, 0, 0, 0, 0, 0
# DIS: 14: {{.*}}ttcleardvalid 0, 1
# DIS: 18: {{.*}}ttsetrwc 2, 0, 0, 0, 0, 0
# DIS: 1c: {{.*}}ttcleardvalid 0, 2
# INVALID-COUNT-5: error: Wormhole UNPACR_NOP does not match an admitted logical form
# INVALID: error: Wormhole CLEARDVALID field reset must be in [0, 0]
# INVALID: error: Wormhole CLEARDVALID field reset must be in [0, 0]
# INVALID: error: Wormhole CLEARDVALID field cleardvalid must be in [1, 2]
# INVALID: error: Wormhole CLEARDVALID field cleardvalid must be in [1, 2]
# INVALID: error: Wormhole SETRWC field bitmask must be in [1, 2]
# INVALID: error: Wormhole SETRWC field rwc_a must be in [0, 0]
# INVALID: error: Wormhole SETRWC field clear_ab_vld must be in [0, 0]

#--- valid.s
ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0
ttunpacr_nop 1, 0, 0, 1, 0, 0, 0, 0, 1
ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1
ttsetrwc 1, 0, 0, 0, 0, 0
ttcleardvalid 0, 1
ttsetrwc 2, 0, 0, 0, 0, 0
ttcleardvalid 0, 2

#--- invalid.s
# All five tuples lie inside the per-field rectangular bounds. None is a form.
ttunpacr_nop 0, 0, 0, 0, 0, 0, 0, 0, 0
ttunpacr_nop 1, 0, 0, 1, 0, 1, 0, 0, 0
ttunpacr_nop 1, 0, 0, 0, 0, 0, 0, 0, 0
ttunpacr_nop 0, 0, 0, 1, 0, 0, 0, 0, 0
ttunpacr_nop 1, 0, 0, 0, 0, 1, 0, 0, 0
ttcleardvalid 1, 1
ttcleardvalid 2, 1
ttcleardvalid 0, 0
ttcleardvalid 0, 3
ttsetrwc 3, 0, 0, 0, 0, 0
ttsetrwc 1, 1, 0, 0, 0, 0
ttsetrwc 1, 0, 0, 0, 0, 1
