# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/invalid.s -o %t-bad.o 2>&1 | FileCheck %s --check-prefix=BAD
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved.txt 2>&1 | FileCheck %s --check-prefix=RAW

# WH MOVA2D/B2D/D2A/D2B: DstRow[9:0], AddrMod[16:15], SrcRow[22:17],
# UseDst32bLo[23]. A2D/D2A/D2B select the multi-row form with raw bit 13.
# B2D's column and row broadcast and four-row selection occupy bits 12..14.
# WH: ttmova2d 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x1a,0x4a]
# WH: ttmovb2d 13, 5, 1, 3, 1{{ *}}# encoding: [0x34,0x40,0x1b,0x4e]
# WH: ttmovd2a 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x1a,0x22]
# WH: ttmovd2b 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x1a,0x2a]
# BH: ttmova2d 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x19,0x4a]
# BH: ttmovb2d 13, 5, 1, 3, 1{{ *}}# encoding: [0x34,0xa0,0x19,0x4e]
# BH: ttmovd2a 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x19,0x22]
# BH: ttmovd2b 13, 2, 1, 3, 1{{ *}}# encoding: [0x34,0x80,0x19,0x2a]
# DIS: ttmova2d 13, 2, 1, 3, 1
# DIS-NEXT: {{.*}}ttmovb2d 13, 5, 1, 3, 1
# DIS-NEXT: {{.*}}ttmovd2a 13, 2, 1, 3, 1
# DIS-NEXT: {{.*}}ttmovd2b 13, 2, 1, 3, 1
# BAD: error: Wormhole MOVA2D field dst must be in [0, 1023]
# BAD: error: Wormhole MOVB2D field addr_mode must be in [0, 3]
# BAD: error: {{.*}}MOVD2A{{.*}}
# BAD: error: {{.*}}MOVD2B{{.*}}
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning: invalid instruction encoding
# RAW: ttmova2d 13, 2, 1, 3, 1

#--- valid.s
ttmova2d 13, 2, 1, 3, 1
ttmovb2d 13, 5, 1, 3, 1
ttmovd2a 13, 2, 1, 3, 1
ttmovd2b 13, 2, 1, 3, 1
#--- invalid.s
ttmova2d 1024, 0, 0, 0, 0
ttmovb2d 0, 0, 4, 0, 0
ttmovd2a 0, 1, 0, 0, 0
ttmovd2b 0, 3, 0, 0, 0
#--- reserved.txt
0x34 0xc0 0x1a 0x4a
0x34 0x80 0x1a 0x4a
