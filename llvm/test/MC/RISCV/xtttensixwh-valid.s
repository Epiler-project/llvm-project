# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=NOFEATURE

# Independently checked against WormholeB0 NOP/STALLWAIT/SEM* diagrams and
# BabyRISCV/PushTensixInstruction.md: direct issue is rotl(raw, 2).
# WH has eight semaphore selectors; BH's wider unused fields are not admitted.
addi a0, a0, 1
# CHECK: ttnop{{ *}}# encoding: [0x00,0x00,0x00,0x08]
# DIS: 4: {{.*}}ttnop
# NOFEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
ttnop
# CHECK: ttstallwait 32767, 511{{ *}}# encoding: [0xfe,0xff,0xff,0x8b]
# DIS: 8: {{.*}}ttstallwait 32767, 511
ttstallwait 32767, 511
# CHECK: ttseminit 255, 15, 15{{ *}}# encoding: [0xf2,0x0f,0xfc,0x8f]
# DIS: c: {{.*}}ttseminit 255, 15, 15
ttseminit 255, 15, 15
# CHECK: ttsemget 255{{ *}}# encoding: [0xf2,0x0f,0x00,0x94]
# DIS: 10: {{.*}}ttsemget 255
ttsemget 255
# CHECK: ttsempost 255{{ *}}# encoding: [0xf2,0x0f,0x00,0x90]
# DIS: 14: {{.*}}ttsempost 255
ttsempost 255
# CHECK: ttsemwait 3, 255, 511{{ *}}# encoding: [0xfe,0x0f,0xfe,0x9b]
# DIS: 18: {{.*}}ttsemwait 3, 255, 511
ttsemwait 3, 255, 511
ret
