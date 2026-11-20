# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=BAD

# WH: ttmovb2a 5, 2, 3, 7{{ *}}# encoding: [0x14,0x80,0x3e,0x2c]
# BH: ttmovb2a 5, 2, 3, 7{{ *}}# encoding: [0x14,0x80,0x3b,0x2c]
# DIS: ttmovb2a 5, 2, 3, 7

#--- valid.s
ttmovb2a 5, 2, 3, 7

#--- invalid.s
# BAD: error: {{.*}}MOVB2A{{.*}}
ttmovb2a 5, 1, 3, 7
# BAD: error: Wormhole MOVB2A field srca must be in [0, 63]
ttmovb2a 5, 2, 3, 64
# BAD: error: Wormhole MOVB2A field srcb must be in [0, 63]
ttmovb2a 64, 2, 3, 7
# BAD: error: Wormhole MOVB2A field addr_mode must be in [0, 3]
ttmovb2a 5, 2, 4, 7
