; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=HAZARD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=HAZARD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; VD=15, Mod1=8 takes configuration data from the author's physical LReg0.
; Imm16 selects first-row columns by even bits and the hardware broadcasts
; each selected column vertically. This ABI writes neither LReg0 nor a CReg
; and invents no staging copy, predicate enable, configuration reset or value.
; It does not claim that the incoming LReg0/CC state has any particular value.
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @masked_lane_config() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: masked_lane_config
; ISEL-NOT: class: sfpr
; ISEL-NOT: PseudoTTBound
; ISEL: TTSFPCONFIGLane 21844, 8, implicit-def $tt_config, implicit-def $tt_issue, implicit $tt_l0, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL-NEXT: TTSFPCONFIGLane 21845, 8, implicit-def $tt_config, implicit-def $tt_issue, implicit $tt_l0, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL-NEXT: TTSFPCONFIGLane 0, 8, implicit-def $tt_config, implicit-def $tt_issue, implicit $tt_l0, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: masked_lane_config:
; NATIVE-NOT: .word
; NATIVE-NOT: call
; NATIVE-NOT: sw
; NATIVE: .word 0x455553e2
; NATIVE-NEXT: .word 0x455557e2
; NATIVE-NEXT: .word 0x440003e2
; NATIVE-NEXT: ret
; Independent ISA words: opcode 0x91, Imm16 at bits23:8, VD=15 at bits7:4,
; Mod1=8 at bits3:0. Rotate each raw word left by two for direct issue.
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21845, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 0, i32 8)
  ret void
}

; Configuration reads LReg0 outside the MAD-result forwarding path. The
; existing hazard owner must insert one issue gap, without a repair copy.
define void @lane_config_reads_actual_l0() "tensix-executor"="trisc2" {
; ISEL-LABEL: name: lane_config_reads_actual_l0
; ISEL-NOT: class: sfpr
; ISEL: $tt_l0 = TTSFPMAD $tt_l0, $tt_l1, $tt_l2, $tt_l3, 0,
; ISEL-NEXT: TTSFPCONFIGLane 21844, 8,
; ISEL-NEXT: PseudoRET
; HAZARD-LABEL: name: lane_config_reads_actual_l0
; HAZARD: $tt_l0 = TTSFPMAD $tt_l0, $tt_l1, $tt_l2, $tt_l3, 0,
; HAZARD-NEXT: TTSFPNOP
; HAZARD-NEXT: TTSFPCONFIGLane 21844, 8,
; HAZARD-NEXT: PseudoRET
; NATIVE-LABEL: lane_config_reads_actual_l0:
; NATIVE: .word 0x10048c02
; NATIVE-NEXT: .word 0x3c000002
; NATIVE-NEXT: .word 0x455553e2
; NATIVE-NEXT: ret
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  ret void
}
