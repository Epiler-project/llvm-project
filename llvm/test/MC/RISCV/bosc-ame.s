# RUN: llvm-mc -triple=riscv64 -mattr=+xboscame -show-encoding < %s | FileCheck %s
# RUN: llvm-mc -triple=riscv64 -mattr=+xboscame -filetype=obj %s -o %t
# RUN: llvm-objdump -d --mattr=+xboscame %t | FileCheck %s --check-prefix=DIS

# The encodings are the patch7 AME words used by the bare-metal GEMM image.
# The register names are architectural resources owned by the MLIR receiver.
msettilem zero, a0
# CHECK: msettilem zero, a0 # encoding: [0x77,0x50,0x05,0x04]
msettilen zero, a0
# CHECK: msettilen zero, a0 # encoding: [0x77,0x40,0x05,0x04]
msettilek zero, a0
# CHECK: msettilek zero, a0 # encoding: [0x77,0x60,0x05,0x04]
mlae8.m tr0, (a0), a1
# CHECK: mlae8.m tr0, (a0), a1 # encoding: [0x77,0x00,0xb5,0x04]
mlbe8.m tr1, (a0), a1
# CHECK: mlbe8.m tr1, (a0), a1 # encoding: [0xf7,0x00,0xb5,0x08]
mlce32.m acc0, (a0), a1
# CHECK: mlce32.m acc0, (a0), a1 # encoding: [0x77,0x20,0xb5,0x00]
mqma.b.mm acc0, tr0, tr1
# CHECK: mqma.b.mm acc0, tr0, tr1 # encoding: [0x77,0x08,0x18,0x28]
msce32.m acc0, (a0), a1
# CHECK: msce32.m acc0, (a0), a1 # encoding: [0x77,0x20,0xb5,0x02]

# DIS: msettilem{{[[:space:]]+}}zero, a0
# DIS: msettilen{{[[:space:]]+}}zero, a0
# DIS: msettilek{{[[:space:]]+}}zero, a0
# DIS: mlae8.m{{[[:space:]]+}}tr0, (a0), a1
# DIS: mlbe8.m{{[[:space:]]+}}tr1, (a0), a1
# DIS: mlce32.m{{[[:space:]]+}}acc0, (a0), a1
# DIS: mqma.b.mm{{[[:space:]]+}}acc0, tr0, tr1
# DIS: msce32.m{{[[:space:]]+}}acc0, (a0), a1
