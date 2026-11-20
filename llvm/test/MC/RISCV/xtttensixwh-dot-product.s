# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=BAD

# WH: ttdotpv 24, 3, 0, 0, 3{{ *}}# encoding: [0x60,0x00,0x06,0xa7]
# BH: ttdotpv 24, 3, 0, 0, 3{{ *}}# encoding: [0x60,0x00,0x03,0xa7]
# DIS: ttdotpv 24, 3, 0, 0, 3

#--- valid.s
ttdotpv 24, 3, 0, 0, 3

#--- invalid.s
# BAD: error: Wormhole DOTPV field instr_mod19 must be in [0, 0]
ttdotpv 24, 3, 1, 0, 3
# BAD: error: Wormhole DOTPV field dest_accum_en must be in [0, 0]
ttdotpv 24, 3, 0, 1, 3
# BAD: error: Wormhole DOTPV field addr_mode must be in [0, 3]
ttdotpv 24, 4, 0, 0, 3
# BAD: error: Wormhole DOTPV field dst must be in [0, 1023]
ttdotpv 1024, 3, 0, 0, 3
