# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/mixed.txt | FileCheck %s --check-prefix=MIXED
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/unknown.txt 2>&1 | FileCheck %s --check-prefix=UNKNOWN
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/truncated.txt 2>&1 | FileCheck %s --check-prefix=TRUNCATED
# MIXED: addi a0, a0, 1
# MIXED-NEXT: ttsetc16 2, 1
# MIXED-NEXT: ret
# UNKNOWN: warning: invalid instruction encoding
# UNKNOWN-NOT: ttsetc16
# UNKNOWN: ret
# TRUNCATED: warning: invalid instruction encoding
# TRUNCATED-NOT: ttsetc16

#--- mixed.txt
0x13 0x05 0x15 0x00 0x0a 0x00 0x04 0xc8 0x67 0x80 0x00 0x00
#--- unknown.txt
0x02 0x00 0x10 0xc9 0x67 0x80 0x00 0x00
#--- truncated.txt
0x0a 0x00 0x04
