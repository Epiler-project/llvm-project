# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %s 2>&1 | FileCheck %s

ttseminit 256, 0, 1
# CHECK: error: Wormhole SEMINIT field sem_sel must be in [0, 255]
ttsemget 256
# CHECK: error: Wormhole SEMGET field sem_sel must be in [0, 255]
ttsempost 256
# CHECK: error: Wormhole SEMPOST field sem_sel must be in [0, 255]
ttsemwait 1, 256, 2
# CHECK: error: Wormhole SEMWAIT field sem_sel must be in [0, 255]
ttsemwait 0, 1, 2
# CHECK: error: Wormhole SEMWAIT field wait_sem_cond must be in [1, 3]
# The initial WH feature is deliberately a bounded instruction subset.
ttsetc16 2, 57
# CHECK: error: Wormhole SETC16 field setc16_reg must be in [0, 56]
ttsfpmov l0, l1
# CHECK: error: instruction requires the following: 'XTTTensixBH'
