; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unsafe-replay.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPLAY-REJECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/unsafe-replay.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPLAY-REJECT
;
; A bound SFPU scope retains ordinary native issue, local dynamic instruction
; ports, complete MOP programming and ordinary scalar stack memory. GPR
; scratch and stack operations may be selected and allocated; none may become
; an SFPU temporary or an additional data move. Local port identity belongs to
; the selected TRISC issuer, not to a hard-coded Math wrapper.

;--- valid.ll
declare void @llvm.riscv.tt.setadc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setadc.port(i32 immarg, i32, i32, i32, i32)
declare void @llvm.riscv.tt.setadc.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define i32 @bound_with_ordinary(i8 noundef %field) "tensix-executor"="trisc2" {
; ISEL-LABEL: name: bound_with_ordinary
; ISEL-NOT: class: sfpr
; ISEL: TTSETADC 7, 0, 0, 1,
; ISEL: early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSETADCPort 0,
; ISEL: early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSETADCMop 2, 1, 0, 0, 1,
; ISEL: PseudoTTMOPClear 3,
; ISEL: PseudoTTMOPClear 8,
; ISEL: TTMOP 0, 0, 1,
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NOT: class: sfpr
; ISEL: TTSFPSTORE $tt_l0, 0, 0, 4,
; NATIVE-LABEL: bound_with_ordinary:
; NATIVE: sw {{[a-z0-9]+}}, {{[0-9]+}}(sp)
; NATIVE: .word
; NATIVE: sw
; NATIVE: .word 0x06000000
; Direct issue rotates each raw ISA word left by two bits.
; SFPMOV(all, l0, c9): rol32(0x7c000902, 2) = 0xf0002409.
; SFPSTORE(l0, offset=0, addrmod=0, format=4):
; rol32(0x72040000, 2) = 0xc8100001. These are fixed encoding oracles,
; independent of the LLVM-generated text checked above.
; NATIVE-NEXT: .word 0xf0002409
; NATIVE-NEXT: .word 0xc8100001
; NATIVE: lw {{[a-z0-9]+}}, {{[0-9]+}}(sp)
; NATIVE: ret
entry:
  %slot = alloca i32, align 4
  %value = zext i8 %field to i32
  store volatile i32 %value, ptr %slot, align 4
  call void @llvm.riscv.tt.setadc(i32 7, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.setadc.port(i32 0, i32 %value, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.setadc.mop(i32 2, i32 1, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  %result = load volatile i32, ptr %slot, align 4
  ret i32 %result
}

; A bound function may still own an explicit ordinary recording and execute
; it while an LReg value stays live. The explicit replay owner prevents
; automatic selection, and the final physical value remains in L0.
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()
define void @bound_with_explicit_replay() "tensix-executor"="trisc0" {
; ISEL-LABEL: name: bound_with_explicit_replay
; ISEL-NOT: class: sfpr
; ISEL: $tt_l0 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: TTREPLAY 1, 0, 1, 0,
; ISEL-NEXT: TTNOP
; ISEL-NEXT: PseudoTTReplayRecordEnd
; ISEL-NEXT: TTREPLAY 0, 0, 1, 0,
; ISEL-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: bound_with_explicit_replay:
; NATIVE: .word 0xf0002409
; REPLAY encodes raw opcode 0x04, length at bit 4 and load at bit 0;
; direct issue rotates left by two. TTNOP's raw opcode is 0x02.
; NATIVE-NEXT: .word 0x10000044
; NATIVE-NEXT: .word 0x08000000
; NATIVE-NEXT: .word 0x10000040
; The explicit replay handoff retains its conservative SFPU hazard NOP;
; rol32(0x8f000000, 2) = 0x3c000002. No data movement is added.
; NATIVE-NEXT: .word 0x3c000002
; NATIVE-NEXT: .word 0xc8100001
; NATIVE-NEXT: ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

;--- unsafe-replay.ll
; Explicit physical operands do not by themselves give an authored SFPU
; recording a preservation ABI. Only the separately verified automatic SFPU
; recording owner may admit that body.
; REPLAY-REJECT: SFPU replay recording requires a typed SSA preservation ABI
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
define void @unsafe_bound_replay() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.replay.record.end()
  ret void
}
