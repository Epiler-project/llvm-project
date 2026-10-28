# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/full.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/full.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/bh-upper.s | FileCheck %s --check-prefix=BH-UPPER
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/full.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: not llvm-mc -triple=riscv32 %t/full.s 2>&1 | FileCheck %s --check-prefix=NOFEATURE
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/revert.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved10.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved17.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved22.txt 2>&1 | FileCheck %s --check-prefix=RAW

# WH ZEROACC.md and Bits32_ZEROACC define Mode at raw bits 20:19,
# AddrMod at 16:15, UseDst32b at 21 and Revert at 18. The canonical Full
# forms have where=0, AddrMod=1, clear_zero_flags=0 and Mode=3. The same
# five logical operands remain separate; logical clear_mode is not the
# SDK's combined Mode | (UseDst32b << 2) argument.
# Independently derived raw words are 0x10188000 and 0x10388000. Direct
# issue rotates them left by two. BH keeps its own AddrMod/use32 positions.
# WH: ttzeroacc 0, 1, 0, 0, 3{{ *}}# encoding: [0x00,0x00,0x62,0x40]
# WH: ttzeroacc 0, 1, 0, 1, 3{{ *}}# encoding: [0x00,0x00,0xe2,0x40]
# BH: ttzeroacc 0, 1, 0, 0, 3{{ *}}# encoding: [0x00,0x00,0x61,0x40]
# BH: ttzeroacc 0, 1, 0, 1, 3{{ *}}# encoding: [0x00,0x00,0x71,0x40]
# BH-UPPER: ttzeroacc 16383, 7, 1, 1, 31{{ *}}# encoding: [0xfc,0xff,0xff,0x43]
# DIS: 0: {{.*}}ttzeroacc 0, 1, 0, 0, 3
# DIS-NEXT: 4: {{.*}}ttzeroacc 0, 1, 0, 1, 3
# INVALID: error: Wormhole ZEROACC field where must be in [0, 0]
# INVALID: error: Wormhole ZEROACC field addr_mode must be in [1, 1]
# INVALID: error: Wormhole ZEROACC field addr_mode must be in [1, 1]
# INVALID: error: Wormhole ZEROACC field clear_zero_flags must be in [0, 0]
# INVALID: error: ZEROACC field use_32_bit_mode must be in [0, 1]
# INVALID: error: Wormhole ZEROACC field clear_mode must be in [3, 3]
# INVALID: error: Wormhole ZEROACC field clear_mode must be in [3, 3]
# INVALID: error: ZEROACC field clear_mode must be in [0, 31]
# NOFEATURE: error: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'

# A rejected direct word must consume all four bytes. Each fixture then
# contains a valid FP32 Full word and ret, so an invalid Revert/reserved bit
# cannot be ignored or interpreted as a compressed instruction.
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning: invalid instruction encoding
# RAW-NOT: ttzeroacc
# RAW: ttzeroacc 0, 1, 0, 1, 3
# RAW-NEXT: ret

#--- full.s
ttzeroacc 0, 1, 0, 0, 3
ttzeroacc 0, 1, 0, 1, 3

#--- bh-upper.s
ttzeroacc 16383, 7, 1, 1, 31

#--- invalid.s
# These first four and the mode2/7 forms fit BH's logical field envelope,
# but are not canonical Full forms received by the WH native subset.
ttzeroacc 1, 1, 0, 0, 3
ttzeroacc 0, 0, 0, 0, 3
ttzeroacc 0, 2, 0, 0, 3
ttzeroacc 0, 1, 1, 0, 3
ttzeroacc 0, 1, 0, 2, 3
ttzeroacc 0, 1, 0, 0, 2
ttzeroacc 0, 1, 0, 1, 7
ttzeroacc 0, 1, 0, 0, 32

#--- revert.txt
0x00 0x00 0x72 0x40 0x00 0x00 0xe2 0x40 0x67 0x80 0x00 0x00
#--- reserved10.txt
0x00 0x10 0x62 0x40 0x00 0x00 0xe2 0x40 0x67 0x80 0x00 0x00
#--- reserved17.txt
0x00 0x00 0x6a 0x40 0x00 0x00 0xe2 0x40 0x67 0x80 0x00 0x00
#--- reserved22.txt
0x00 0x00 0x62 0x41 0x00 0x00 0xe2 0x40 0x67 0x80 0x00 0x00
