# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/bh-upper.s | FileCheck %s --check-prefix=BH-UPPER
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/invalid.s -o %t-invalid.o 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: not test -e %t-invalid.o
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved-row10.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved-addr14.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved-addr17.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved-mul-accum.txt 2>&1 | FileCheck %s --check-prefix=RAW

# Independently checked against WH ELWADD.md and Bits32_ELWADD.svg:
# raw opcode0x28; row9:0, AddrMod16:15, broadcast20:19, AddDst21,
# flips23:22. Direct issue rotates raw left2. BH's existing AddrMod
# begins at raw14; its original wider native logical ABI remains unchanged.
# WH: ttelwadd 0, 0, 0, 0, 0{{ *}}# encoding: [0x00,0x00,0x00,0xa0]
# WH: ttelwadd 0, 1, 0, 0, 0{{ *}}# encoding: [0x00,0x00,0x02,0xa0]
# WH: ttelwadd 1023, 3, 3, 1, 3{{ *}}# encoding: [0xfc,0x0f,0xe6,0xa3]
# WH: ttelwsub 24, 3, 2, 1, 3{{ *}}# encoding: [0x60,0x00,0xc6,0xc3]
# WH: ttelwmul 24, 3, 2, 0, 3{{ *}}# encoding: [0x60,0x00,0x46,0x9f]
# BH: ttelwadd 0, 0, 0, 0, 0{{ *}}# encoding: [0x00,0x00,0x00,0xa0]
# BH: ttelwadd 0, 1, 0, 0, 0{{ *}}# encoding: [0x00,0x00,0x01,0xa0]
# BH: ttelwadd 1023, 3, 3, 1, 3{{ *}}# encoding: [0xfc,0x0f,0xe3,0xa3]
# BH: ttelwsub 24, 3, 2, 1, 3{{ *}}# encoding: [0x60,0x00,0xc3,0xc3]
# BH: ttelwmul 24, 3, 2, 0, 3{{ *}}# encoding: [0x60,0x00,0x43,0x9f]
# BH-UPPER: ttelwadd 16383, 31, 3, 1, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xa3]
# DIS: ttelwadd 0, 0, 0, 0, 0
# DIS-NEXT: {{.*}}ttelwadd 0, 1, 0, 0, 0
# DIS-NEXT: {{.*}}ttelwadd 1023, 3, 3, 1, 3
# DIS-NEXT: {{.*}}ttelwsub 24, 3, 2, 1, 3
# DIS-NEXT: {{.*}}ttelwmul 24, 3, 2, 0, 3
# INVALID: error: Wormhole ELWADD field dst must be in [0, 1023]
# INVALID: error: Wormhole ELWADD field addr_mode must be in [0, 3]
# INVALID: error: ELWADD field instr_mod19 must be in [0, 3]
# INVALID: error: ELWADD field dest_accum_en must be in [0, 1]
# INVALID: error: ELWADD field clear_dvalid must be in [0, 3]
# INVALID: error: Wormhole ELWSUB field addr_mode must be in [0, 3]
# INVALID: error: Wormhole ELWMUL field dest_accum_en must be in [0, 0]
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning: invalid instruction encoding
# RAW-NOT: ttelwadd
# RAW-NOT: ttelwmul
# RAW: ttelwadd 0, 1, 0, 0, 0
# RAW-NEXT: ret

#--- valid.s
ttelwadd 0, 0, 0, 0, 0
ttelwadd 0, 1, 0, 0, 0
ttelwadd 1023, 3, 3, 1, 3
ttelwsub 24, 3, 2, 1, 3
ttelwmul 24, 3, 2, 0, 3

#--- bh-upper.s
ttelwadd 16383, 31, 3, 1, 3

#--- invalid.s
ttelwadd 1024, 0, 0, 0, 0
ttelwadd 0, 4, 0, 0, 0
ttelwadd 0, 0, 4, 0, 0
ttelwadd 0, 0, 0, 2, 0
ttelwadd 0, 0, 0, 0, 4
ttelwsub 24, 4, 2, 1, 3
ttelwmul 24, 3, 2, 1, 3

#--- reserved-row10.txt
0x00 0x10 0x00 0xa0
0x00 0x00 0x02 0xa0
0x67 0x80 0x00 0x00

#--- reserved-addr14.txt
0x00 0x00 0x01 0xa0
0x00 0x00 0x02 0xa0
0x67 0x80 0x00 0x00

#--- reserved-addr17.txt
0x00 0x00 0x08 0xa0
0x00 0x00 0x02 0xa0
0x67 0x80 0x00 0x00

#--- reserved-mul-accum.txt
0x60 0x00 0xc6 0x9f
0x00 0x00 0x02 0xa0
0x67 0x80 0x00 0x00
