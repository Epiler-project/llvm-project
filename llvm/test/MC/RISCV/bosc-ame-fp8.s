# RUN: llvm-mc -triple riscv64 -mattr=+xboscame -show-encoding < %s | FileCheck %s
# AME v0.5 documented words: 0x020b6077 and 0x2a180877.
# CHECK: msetfp8.e4m3 zero{{.*}}encoding: [0x77,0x60,0x0b,0x02]
msetfp8.e4m3 zero
# CHECK: mfqma.cf.mm acc0, tr0, tr1{{.*}}encoding: [0x77,0x08,0x18,0x2a]
mfqma.cf.mm acc0, tr0, tr1
