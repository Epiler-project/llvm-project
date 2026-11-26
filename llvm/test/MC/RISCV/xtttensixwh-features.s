# RUN: not llvm-mc -triple=riscv64 -mattr=+xtttensixwh %s 2>&1 | FileCheck %s --check-prefix=RV32
# RUN: not llvm-mc -triple=riscv32be -mattr=+xtttensixwh %s 2>&1 | FileCheck %s --check-prefix=RV32
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh,+c %s 2>&1 | FileCheck %s --check-prefix=COMPRESSED
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh,+zcb %s 2>&1 | FileCheck %s --check-prefix=COMPRESSED
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh,+v %s 2>&1 | FileCheck %s --check-prefix=VECTOR
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh,+zve64d %s 2>&1 | FileCheck %s --check-prefix=VECTOR
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh,+xtttensixbh %s 2>&1 | FileCheck %s --check-prefix=MIXED
# RV32: XTTTensixWH requires RV32 little-endian
# COMPRESSED: XTTTensixWH is incompatible with C/Zc extensions
# VECTOR: XTTTensixWH is incompatible with V/Zve extensions
# MIXED: XTTTensixBH and XTTTensixWH are mutually exclusive
ttnop
