# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/fields.s 2>&1 | FileCheck %s --check-prefix=FIELDS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/bh.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/bh.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=BH-DIS
# RUN: not llvm-mc -triple=riscv32 %t/bh.s 2>&1 | FileCheck %s --check-prefix=FEATURE
# FIELDS: error:
# FIELDS: error: immediate offset must be in the range [0, 511]
# FIELDS: error: immediate must be an integer in the range [0, 3]
# FIELDS: error: immediate must be an integer in the range [0, 4095]
# FIELDS: error: Wormhole RMWCIB0 does not match an admitted logical form
# BH: ttsfploadmacro.fp32.even l0, 8, 0{{ *}}# encoding: [0x42,0x00,0x0c,0x4c]
# BH-DIS: ttsfploadmacro.fp32.even l0, 8, 0
# FEATURE: error: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
#--- fields.s
ttsfploadmacro.fp32.even l4, 8, 0
ttsfploadmacro.fp32.even l0, 512, 0
ttsfpconfig.macro.template 4
ttsfpconfig.macro.misc 4096
ttrmwcib0 6, 1, 255
#--- bh.s
ttsfploadmacro.fp32.even l0, 8, 0
