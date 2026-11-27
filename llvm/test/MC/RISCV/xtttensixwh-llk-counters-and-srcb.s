# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/invalid.s | FileCheck %s --check-prefix=BH

#--- valid.s
ttaddrcrxy 15, 7, 7, 7, 7, 7
ttaddrcrzw 1, 0, 0, 0, 0, 1
ttincadcxy 0, 1, 0, 0, 1
ttincadczw 1, 0, 0, 0, 2
ttincrwc 15, 15, 15, 7
ttsetadcxx 0, 255, 1
ttsetadcxy 11, 0, 0, 0, 0, 3
ttsetadczw 15, 1, 0, 2, 0, 1
tttrnspsrcb

# CHECK: ttaddrcrxy 15, 7, 7, 7, 7, 7{{ *}}# encoding: [0x3d,0xff,0x8f,0x4f]
# CHECK: ttaddrcrzw 1, 0, 0, 0, 0, 1{{ *}}# encoding: [0x05,0x00,0x80,0x58]
# CHECK: ttincadcxy 0, 1, 0, 0, 1{{ *}}# encoding: [0x01,0x08,0x80,0x48]
# CHECK: ttincadczw 1, 0, 0, 0, 2{{ *}}# encoding: [0x01,0x01,0x00,0x55]
# CHECK: ttincrwc 15, 15, 15, 7{{ *}}# encoding: [0x00,0xff,0x7f,0xe0]
# CHECK: ttsetadcxx 0, 255, 1{{ *}}# encoding: [0x01,0xf0,0x8f,0x78]
# CHECK: ttsetadcxy 11, 0, 0, 0, 0, 3{{ *}}# encoding: [0x2d,0x00,0x80,0x45]
# CHECK: ttsetadczw 15, 1, 0, 2, 0, 1{{ *}}# encoding: [0x3d,0x81,0x80,0x50]
# CHECK: tttrnspsrcb{{ *}}# encoding: [0x00,0x00,0x00,0x58]

# DIS: ttaddrcrxy 15, 7, 7, 7, 7, 7
# DIS-NEXT: {{.*}}ttaddrcrzw 1, 0, 0, 0, 0, 1
# DIS-NEXT: {{.*}}ttincadcxy 0, 1, 0, 0, 1
# DIS-NEXT: {{.*}}ttincadczw 1, 0, 0, 0, 2
# DIS-NEXT: {{.*}}ttincrwc 15, 15, 15, 7
# DIS-NEXT: {{.*}}ttsetadcxx 0, 255, 1
# DIS-NEXT: {{.*}}ttsetadcxy 11, 0, 0, 0, 0, 3
# DIS-NEXT: {{.*}}ttsetadczw 15, 1, 0, 2, 0, 1
# DIS-NEXT: {{.*}}tttrnspsrcb

#--- invalid.s
ttaddrcrxy 16, 0, 0, 0, 0, 1
ttsetadcxy 1, 0, 0, 0, 8, 1
ttincadczw 0, 0, 0, 8, 1
ttincrwc 0, 0, 0, 8
ttsetadcxx 0, 1024, 4

# INVALID: error: Wormhole ADDRCRXY field bitmask must be in [0, 15]
# INVALID: error: Wormhole SETADCXY field ch1_y must be in [0, 7]
# INVALID: error: Wormhole INCADCZW field ch1_y must be in [0, 7]
# INVALID: error: Wormhole INCRWC field rwc_cr must be in [0, 7]
# INVALID: error: Wormhole SETADCXX field x_end2 must be in [0, 1023]
# BH: ttaddrcrxy 16, 0, 0, 0, 0, 1
# BH: ttsetadcxy 1, 0, 0, 0, 8, 1
# BH: ttincadczw 0, 0, 0, 8, 1
# BH: ttincrwc 0, 0, 0, 8
# BH: ttsetadcxx 0, 1024, 4
