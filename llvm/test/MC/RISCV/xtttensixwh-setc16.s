# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/common.s | FileCheck %s --check-prefix=COMMON
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/common.s | FileCheck %s --check-prefix=COMMON
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/bh-upper.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/common.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

# SETC16.md and pinned WH ckernel_ops.h define RawWord = 0xb2000000 |
# (CfgIndex << 16) | NewValue. The RISC-V carrier rotates this word by two.
# WH cfg_defines.h publishes THD_STATE_SIZE=57, so CfgIndex 57 is invalid.
# Blackhole retains its own 68-entry thread configuration range.
# COMMON: ttsetc16 0, 0{{ *}}# encoding: [0x02,0x00,0x00,0xc8]
# COMMON: ttsetc16 65535, 56{{ *}}# encoding: [0xfe,0xff,0xe3,0xc8]
# COMMON: ttsetc16 0, 4{{ *}}# encoding: [0x02,0x00,0x10,0xc8]
# COMMON: ttsetc16 0, 39{{ *}}# encoding: [0x02,0x00,0x9c,0xc8]
# BH: ttsetc16 65535, 67{{ *}}# encoding: [0xfe,0xff,0x0f,0xc9]
# DIS: 0: {{.*}}ttsetc16 0, 0
# DIS: 4: {{.*}}ttsetc16 65535, 56
# DIS: 8: {{.*}}ttsetc16 0, 4
# DIS: c: {{.*}}ttsetc16 0, 39
# INVALID-COUNT-2: error: Wormhole SETC16 field setc16_reg must be in [0, 56]
# INVALID: error: SETC16 field setc16_value must be in [0, 65535]

#--- common.s
ttsetc16 0, 0
ttsetc16 65535, 56
ttsetc16 0, 4
ttsetc16 0, 39

#--- bh-upper.s
ttsetc16 65535, 67

#--- invalid.s
ttsetc16 0, 57
ttsetc16 0, 67
ttsetc16 65536, 0
