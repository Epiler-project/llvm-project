; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/bound.ll -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/bound.ll -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/bound.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/bound.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/bound.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=MC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/bound.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=MC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/legacy.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=LEGACY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/legacy.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=LEGACY
;
; Each load is a partial read-modify-write of the same real old LReg. Keep
; the old operand and CC/config/Dst/issue effects through both lowering routes.
; This verifies faithful encoding, not initialized Dst or lane setup on device.
; No replay recipe or source-loop expansion is involved.
;--- bound.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @bound_partial_words() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_partial_words
; ISEL: $tt_l3 = TTSFPMOVAll $tt_c8,
; ISEL-NEXT: $tt_l3 = TTSFPLOAD $tt_l3, 513, 7, 14,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: TTSFPSTORE $tt_l3, 1023, 7, 14,
; ISEL-NEXT: $tt_l3 = TTSFPLOAD $tt_l3, 514, 7, 15,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: TTSFPSTORE $tt_l3, 1022, 7, 15,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: bound_partial_words:
; NATIVE: .word 0xf00020c9
; NATIVE-NEXT: .word 0xc0fb8805
; NATIVE-NEXT: .word 0xc8fb8ffd
; NATIVE-NEXT: .word 0xc0ff8809
; NATIVE-NEXT: .word 0xc8ff8ff9
; NATIVE-NEXT: ret
; MC-LABEL: <bound_partial_words>:
; MC: ttsfpmov.all l3, c8
; MC-NEXT: {{.*}}ttsfpload l3, 513, 7, 14
; MC-NEXT: {{.*}}ttsfpstore l3, 1023, 7, 14
; MC-NEXT: {{.*}}ttsfpload l3, 514, 7, 15
; MC-NEXT: {{.*}}ttsfpstore l3, 1022, 7, 15
; MC-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 8)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 513, i32 7, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 1023, i32 7, i32 14)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 514, i32 7, i32 15)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 1022, i32 7, i32 15)
  ret void
}

define void @bound_partial_dynamic(i10 noundef %index) "tensix-executor"="trisc2" {
; ISEL-LABEL: name: bound_partial_dynamic
; ISEL: $tt_l6 = TTSFPMOVAll $tt_c8,
; ISEL-NEXT: $tt_l6, early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPLOAD $tt_l6, %{{[0-9]+}}, 7, 14,
; ISEL-NEXT: early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPSTORE $tt_l6, %{{[0-9]+}}, 7, 15,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: bound_partial_dynamic:
; NATIVE: sw
; NATIVE: sw
; NATIVE: ret
; MC-LABEL: <bound_partial_dynamic>:
; MC: ttsfpmov.all l6, c8
; MC: sw
; MC: sw
; MC: ret
  %offset = zext i10 %index to i32
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 8)
  call void @llvm.riscv.tt.bound.sfpload(i32 6, i32 6, i32 %offset, i32 7, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 %offset, i32 7, i32 15)
  ret void
}

;--- legacy.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @legacy_partial_words() "tensix-executor"="trisc1" {
; LEGACY-LABEL: <legacy_partial_words>:
; LEGACY: ttsfpload [[LREG:l[0-7]]], 0, 7, 14
; LEGACY: ttsfpstore [[LREG]], 2, 7, 14
; LEGACY: ttsfpload [[LREG]], 4, 7, 15
; LEGACY: ttsfpstore [[LREG]], 6, 7, 15
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 8)
  %low = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 0, i32 7, i32 14)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %low, i32 2, i32 7, i32 14)
  %both = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %low, i32 4, i32 7, i32 15)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %both, i32 6, i32 7, i32 15)
  ret void
}
