# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/rvc.s 2>&1 | FileCheck %s --check-prefix=COMPRESSED
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/arch.s 2>&1 | FileCheck %s --check-prefix=VECTOR
# COMPRESSED: XTTTensixBH is incompatible with C/Zc extensions
# VECTOR: XTTTensixBH is incompatible with V/Zve extensions

# Feature legality also applies after assembler directives mutate the STI.
#--- rvc.s
.option rvc
addi a0, a0, 1
#--- arch.s
.option arch, +v
ttsetc16 2, 1
