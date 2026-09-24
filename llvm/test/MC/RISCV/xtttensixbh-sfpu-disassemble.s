# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/mixed.txt | FileCheck %s --check-prefix=MIXED
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/bad-register.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/bad-loadi-mode.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/bad-encc-mode.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/bad-dst-format.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/bad-dst-offset.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/truncated.txt 2>&1 | FileCheck %s --check-prefix=TRUNCATED
# MIXED: addi a0, a0, 1
# MIXED-NEXT: ttsfpmov.all l7, c15
# MIXED-NEXT: ttsfpiadd.sub l3, l4
# MIXED-NEXT: ttsfpnop
# MIXED-NEXT: ret
# INVALID: warning: invalid instruction encoding
# INVALID-NOT: ttsfp
# INVALID: ret
# TRUNCATED: warning: invalid instruction encoding
# TRUNCATED-NOT: ttsfp

# The invalid instruction consumes exactly four bytes, leaving the following
# RV32 return intact. A truncated SFPU word must not be decoded as RVC.
#--- mixed.txt
0x13 0x05 0x15 0x00 0xc9 0x3d 0x00 0xf0 0xd9 0x10 0x00 0xe4 0x02 0x00 0x00 0x3c 0x67 0x80 0x00 0x00
#--- bad-register.txt
0x09 0x02 0x00 0xf0 0x67 0x80 0x00 0x00
#--- bad-loadi-mode.txt
0x01 0x00 0x0c 0xc4 0x67 0x80 0x00 0x00
#--- bad-encc-mode.txt
0x0e 0x00 0x00 0x28 0x67 0x80 0x00 0x00
#--- bad-dst-format.txt
0x01 0x00 0x04 0xc0 0x67 0x80 0x00 0x00
#--- bad-dst-offset.txt
0x01 0x10 0x08 0xc0 0x67 0x80 0x00 0x00
#--- truncated.txt
0xc9 0x3d 0x00
