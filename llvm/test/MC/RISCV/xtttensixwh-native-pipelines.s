# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/bh-upper.s | FileCheck %s --check-prefix=BH-UPPER
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/invalid.s -o %t-invalid.o 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: not test -e %t-invalid.o
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/mvmul-reserved10.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/mvmul-reserved14.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/mvmul-reserved17.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/zerosrc-reserved5.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/unpacr-unreceived-context.txt 2>&1 | FileCheck %s --check-prefix=RAW

# WH MVMUL.md: DstRow at 9:0, AddrMod at 16:15, Broadcast at 19,
# FlipSrcA/B at 22/23. In particular AddrMod=1 is raw 0x26008000,
# not the BH logical encoding's raw 0x26004000. Direct issue rotates
# the independently derived raw words left by two.
# UNPACR regular forms follow UNPACR_Regular.md and pinned WH LLK
# llk_unpack_AB_matmul.h: multi-context override, zero context selectors,
# no row search, context-counter increment, cache flush, or rarefy mode.
# Last=1 is the LLK production form; Last=0 is also specified regular syntax.
# ZEROSRC.md has one boolean negative-infinity bit, not BH's 20-bit zero_val.
# WH: ttmvmul 0, 1, 0, 0{{ *}}# encoding: [0x00,0x00,0x02,0x98]
# WH: ttmvmul 1023, 3, 0, 3{{ *}}# encoding: [0xfc,0x0f,0x06,0x9b]
# WH: ttmvmul 16, 2, 1, 2{{ *}}# encoding: [0x40,0x00,0x24,0x9a]
# WH: ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x05,0x03,0x00,0x08]
# WH: ttunpacr 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1{{ *}}# encoding: [0x05,0x02,0x22,0x0a]
# WH: ttunpacr 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 255, 1{{ *}}# encoding: [0x45,0x03,0xfe,0x0b]
# WH: ttunpacr 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x01,0x02,0x00,0x08]
# WH: ttzerosrc 3, 1, 0, 0{{ *}}# encoding: [0x1c,0x00,0x00,0x44]
# WH: ttzerosrc 1, 0, 1, 1{{ *}}# encoding: [0x64,0x00,0x00,0x44]
# WH: ttzerosrc 3, 1, 1, 1{{ *}}# encoding: [0x7c,0x00,0x00,0x44]
# BH: ttmvmul 0, 1, 0, 0{{ *}}# encoding: [0x00,0x00,0x01,0x98]
# BH: ttmvmul 1023, 3, 0, 3{{ *}}# encoding: [0xfc,0x0f,0x03,0x9b]
# BH: ttmvmul 16, 2, 1, 2{{ *}}# encoding: [0x40,0x00,0x22,0x9a]
# BH-UPPER: ttmvmul 16383, 31, 7, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x9b]
# BH-UPPER: ttzerosrc 3, 1, 1, 1048575{{ *}}# encoding: [0xfc,0xff,0xff,0x47]
# DIS: ttmvmul 0, 1, 0, 0
# DIS: ttmvmul 1023, 3, 0, 3
# DIS: ttmvmul 16, 2, 1, 2
# DIS: ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
# DIS: ttunpacr 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1
# DIS: ttunpacr 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 255, 1
# DIS: ttunpacr 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
# DIS: ttzerosrc 3, 1, 0, 0
# DIS: ttzerosrc 1, 0, 1, 1
# DIS: ttzerosrc 3, 1, 1, 1
# INVALID: error: Wormhole MVMUL field dst must be in [0, 1023]
# INVALID: error: Wormhole MVMUL field addr_mode must be in [0, 3]
# INVALID: error: Wormhole MVMUL field instr_mod19 must be in [0, 1]
# INVALID: error: MVMUL field clear_dvalid must be in [0, 3]
# INVALID: error: Wormhole UNPACR field searchcacheflush must be in [0, 0]
# INVALID: error: Wormhole UNPACR field rowsearch must be in [0, 0]
# INVALID: error: Wormhole UNPACR field autoinccontextid must be in [0, 0]
# INVALID: error: Wormhole UNPACR field srcb_bcast must be in [0, 0]
# INVALID: error: Wormhole UNPACR field ovrdthreadid must be in [1, 1]
# INVALID: error: Wormhole UNPACR field addrcntcontextid must be in [0, 0]
# INVALID: error: Wormhole UNPACR field cfgcontextid must be in [0, 0]
# INVALID: error: Wormhole UNPACR field cfgcontextcntinc must be in [0, 0]
# INVALID: error: UNPACR field addrmode must be in [0, 255]
# INVALID: error: UNPACR field unpack_block_selection must be in [0, 1]
# INVALID: error: ZEROSRC field src_mask must be in [0, 3]
# INVALID: error: ZEROSRC field bank_mask must be in [0, 1]
# INVALID: error: ZEROSRC field write_mode must be in [0, 1]
# INVALID: error: Wormhole ZEROSRC field zero_val must be in [0, 1]
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning: invalid instruction encoding
# RAW-NOT: ttmvmul
# RAW-NOT: ttunpacr
# RAW-NOT: ttzerosrc
# RAW: ttmvmul 0, 1, 0, 0
# RAW-NEXT: ret

#--- valid.s
ttmvmul 0, 1, 0, 0
ttmvmul 1023, 3, 0, 3
ttmvmul 16, 2, 1, 2
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1
ttunpacr 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 255, 1
ttunpacr 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
ttzerosrc 3, 1, 0, 0
ttzerosrc 1, 0, 1, 1
ttzerosrc 3, 1, 1, 1

#--- bh-upper.s
ttmvmul 16383, 31, 7, 3
ttzerosrc 3, 1, 1, 1048575

#--- invalid.s
ttmvmul 1024, 0, 0, 0
ttmvmul 0, 4, 0, 0
ttmvmul 0, 0, 2, 0
ttmvmul 0, 0, 0, 4
ttunpacr 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
ttunpacr 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
ttunpacr 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 256, 0
ttunpacr 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2
ttzerosrc 4, 0, 0, 0
ttzerosrc 1, 2, 0, 0
ttzerosrc 1, 0, 2, 0
ttzerosrc 1, 0, 0, 2

#--- mvmul-reserved10.txt
0x00 0x10 0x02 0x98 0x00 0x00 0x02 0x98 0x67 0x80 0x00 0x00
#--- mvmul-reserved14.txt
0x00 0x00 0x03 0x98 0x00 0x00 0x02 0x98 0x67 0x80 0x00 0x00
#--- mvmul-reserved17.txt
0x00 0x00 0x0a 0x98 0x00 0x00 0x02 0x98 0x67 0x80 0x00 0x00
#--- zerosrc-reserved5.txt
0x9c 0x00 0x00 0x44 0x00 0x00 0x02 0x98 0x67 0x80 0x00 0x00
#--- unpacr-unreceived-context.txt
0x05 0x01 0x00 0x08 0x00 0x00 0x02 0x98 0x67 0x80 0x00 0x00
