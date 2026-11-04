# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# Native offset is Imm9; actual address is 2*Imm9 after verified setup.
# CHECK: ttsfploadmacro.fp32.even l0, 8, 0{{ *}}# encoding: [0x42,0x00,0x0c,0x4c]
# DIS: ttsfploadmacro.fp32.even l0, 8, 0
ttsfploadmacro.fp32.even l0, 8, 0
# CHECK: ttsfploadmacro.fp32.even l3, 511, 3{{ *}}# encoding: [0xfa,0x0f,0xcc,0x4f]
# DIS: ttsfploadmacro.fp32.even l3, 511, 3
ttsfploadmacro.fp32.even l3, 511, 3
# CHECK: ttsfpconfig.macro.template 3{{ *}}# encoding: [0xc2,0x00,0x00,0x44]
# DIS: ttsfpconfig.macro.template 3
ttsfpconfig.macro.template 3
# CHECK: ttsfpconfig.macro.sequence 3{{ *}}# encoding: [0xc2,0x01,0x00,0x44]
# DIS: ttsfpconfig.macro.sequence 3
ttsfpconfig.macro.sequence 3
# CHECK: ttsfpconfig.macro.misc 3{{ *}}# encoding: [0x06,0x0e,0x00,0x44]
# DIS: ttsfpconfig.macro.misc 3
ttsfpconfig.macro.misc 3
# CHECK: ttrmwcib0 6, 0, 255{{ *}}# encoding: [0x1a,0x00,0xfc,0xcf]
# DIS: ttrmwcib0 6, 0, 255
ttrmwcib0 6, 0, 255
# CHECK: ttrmwcib1 6, 0, 255{{ *}}# encoding: [0x1a,0x00,0xfc,0xd3]
# DIS: ttrmwcib1 6, 0, 255
ttrmwcib1 6, 0, 255
