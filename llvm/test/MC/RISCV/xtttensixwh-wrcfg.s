# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

#--- valid.s
ttwrcfg 8, 0, 62
ttwrcfg 152, 0, 0
ttwrcfg 187, 0, 63
# CHECK: ttwrcfg 8, 0, 62{{ *}}# encoding: [0x22,0x00,0xf8,0xc0]
# CHECK: ttwrcfg 152, 0, 0{{ *}}# encoding: [0x62,0x02,0x00,0xc0]
# CHECK: ttwrcfg 187, 0, 63{{ *}}# encoding: [0xee,0x02,0xfc,0xc0]

#--- invalid.s
ttwrcfg 188, 0, 0
ttwrcfg 8, 1, 0
ttwrcfg 8, 0, 64
# INVALID: error: {{.*}}cfgreg{{.*}}187
# INVALID: error: {{.*}}wr128b{{.*}}0
# INVALID: error: {{.*}}gpraddress{{.*}}63
