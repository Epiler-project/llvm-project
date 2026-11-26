# RUN: not llvm-mc -triple=riscv64 -mattr=+xtttensixbh %s 2>&1 | FileCheck %s --check-prefix=RV32
# RUN: not llvm-mc -triple=riscv32be -mattr=+xtttensixbh %s 2>&1 | FileCheck %s --check-prefix=RV32
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh,+c %s 2>&1 | FileCheck %s --check-prefix=COMPRESSED
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh,+zcb %s 2>&1 | FileCheck %s --check-prefix=COMPRESSED
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh,+v %s 2>&1 | FileCheck %s --check-prefix=VECTOR
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh,+zve64d %s 2>&1 | FileCheck %s --check-prefix=VECTOR
# RV32: XTTTensixBH requires RV32 little-endian
# COMPRESSED: XTTTensixBH is incompatible with C/Zc extensions
# VECTOR: XTTTensixBH is incompatible with V/Zve extensions
ttsetc16 2, 1
