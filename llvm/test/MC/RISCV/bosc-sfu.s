# RUN: llvm-mc -triple=riscv64 -mattr=+v,+xboscsfu -show-encoding < %s | FileCheck %s
# RUN: llvm-mc -triple=riscv64 -mattr=+v,+xboscsfu -filetype=obj %s -o %t
# RUN: llvm-objdump -d --mattr=+v,+xboscsfu %t | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv64 -mattr=+v %s 2>&1 | FileCheck %s --check-prefix=FEATURE

# RTL VDecoder selects Custom-1/F6=0x13/marker=7; VLane forwards F3 to
# VecSfu: operation zero is exp2 and operation two is reciprocal. vm=1.
vsfu.exp2.v v2, v1
# CHECK: vsfu.exp2.v v2, v1 # encoding: [0x2b,0x81,0x13,0x4e]
vsfu.rcp.v v31, v30
# CHECK: vsfu.rcp.v v31, v30 # encoding: [0xab,0xaf,0xe3,0x4f]
# DIS: vsfu.exp2.v{{[[:space:]]+}}v2, v1
# DIS: vsfu.rcp.v{{[[:space:]]+}}v31, v30
# FEATURE: instruction requires the following: 'XBOSCSFU' (BOSC vector special function extension)
