# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID

#--- valid.s
ttrdcfg 0, 0
ttrdcfg 187, 63
# CHECK: ttrdcfg 0, 0{{ *}}# encoding: [0x02,0x00,0x00,0xc4]
# CHECK: ttrdcfg 187, 63{{ *}}# encoding: [0xee,0x02,0xfc,0xc4]

#--- invalid.s
ttrdcfg 188, 0
ttrdcfg 0, 64
# INVALID: error: {{.*}}cfgreg{{.*}}187
# INVALID: error: {{.*}}gpraddress{{.*}}63
