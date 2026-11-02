# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s -o %t.o
# RUN: llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex %t.o | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/invalid.s -o %t-invalid.o 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: not test -e %t-invalid.o

# CHECK: ttsetrwc 15, 0, 0, 0, 0, 0{{ *}}# encoding: [0x3c,0x00,0x00,0xdc]
# CHECK: ttsetadc 63, 0, 1, 4{{ *}}# encoding: [0xfd,0x00,0x40,0x42]
# CHECK: ttpacr 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0{{ *}}# encoding: [0x0d,0x04,0x00,0x04]
# CHECK: ttpacr 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0{{ *}}# encoding: [0x05,0x04,0x00,0x04]
# DIS: ttsetrwc 15, 0, 0, 0, 0, 0
# DIS: ttsetadc 63, 0, 1, 4
# DIS: ttpacr 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
# DIS: ttpacr 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
# INVALID-COUNT-6: error:

#--- valid.s
ttsetrwc 15, 0, 0, 0, 0, 0
ttsetadc 63, 0, 1, 4
ttpacr 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
ttpacr 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0

#--- invalid.s
# WH does not receive the BH-only context and row padding fields.
ttpacr 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0
ttpacr 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0
ttpacr 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0
# Packer mask5 is an undefined hardware combination.
ttpacr 1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0
# Thread override and bank-flip reset are outside the admitted contract.
ttsetadc 65536, 0, 1, 4
ttsetrwc 15, 0, 0, 0, 0, 1
