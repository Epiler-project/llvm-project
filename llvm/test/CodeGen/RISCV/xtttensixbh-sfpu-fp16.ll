; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/bound.ll -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/bound.ll -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/bound.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/bound.ll -o - | FileCheck %s --check-prefix=NATIVE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/bound.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=MC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/bound.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=MC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/legacy.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETIRED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/legacy.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETIRED
;
; Format 1 is the architectural FP16 Dst conversion with ordinary addressing.
; It must survive the existing bound intrinsic, physical ISel, and MC paths.
; This is a form/encoding test: no host half conversion models its numerical
; behavior, and no new intrinsic, SFPU temporary, or repair copy is needed.
; The separate legacy module tests its still-live verifier's matching format
; admission; it is not a fallback for the bound module.
; RETIRED: unsupported Tensix intrinsic ABI:

;--- bound.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @bound_fp16() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_fp16
; ISEL: $tt_l3 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: $tt_l3 = TTSFPLOAD $tt_l3, 513, 2, 1,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: TTSFPSTORE $tt_l3, 1023, 7, 1,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: bound_fp16:
; NATIVE-NOT: sw
; NATIVE-NOT: lw
; NATIVE: .word 0xf00024c9
; Fixed raw ISA oracles, rotated left two bits for direct issue:
; SFPLOAD  0x70314201 -> 0xc0c50805 (L3, offset 513, AddrMod 2, FP16).
; SFPSTORE 0x7231e3ff -> 0xc8c78ffd (L3, offset 1023, AddrMod 7, FP16).
; NATIVE-NEXT: .word 0xc0c50805
; NATIVE-NEXT: .word 0xc8c78ffd
; NATIVE-NEXT: ret
; MC-LABEL: <bound_fp16>:
; MC: ttsfpmov.all l3, c9
; MC-NEXT: {{.*}}ttsfpload l3, 513, 2, 1
; MC-NEXT: {{.*}}ttsfpstore l3, 1023, 7, 1
; MC-NEXT: {{.*}}ret
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 9)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 513, i32 2, i32 1)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 1023, i32 7, i32 1)
  ret void
}

; A bounded scalar offset uses ordinary GPR scratch for instruction issue;
; its SFPU destination and tied old value stay in the same physical L6.
define void @bound_fp16_dynamic(i10 noundef %index) "tensix-executor"="trisc2" {
; ISEL-LABEL: name: bound_fp16_dynamic
; ISEL: $tt_l6 = TTSFPMOVAll $tt_c9,
; ISEL-NEXT: $tt_l6, early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPLOAD $tt_l6, %{{[0-9]+}}, 7, 1,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPSTORE $tt_l6, %{{[0-9]+}}, 7, 1,
; ISEL-SAME: implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_dst, implicit $tt_issue
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: bound_fp16_dynamic:
; NATIVE: .word 0xf0002589
; Raw dynamic-port bases retain format 1 before the offset is inserted:
; SFPLOAD 0x7061e000 / 4096 = 460318; SFPSTORE 0x7261e000 / 4096 = 468510.
; NATIVE: lui {{[a-z0-9]+}}, 460318
; NATIVE: sw
; NATIVE: lui {{[a-z0-9]+}}, 468510
; NATIVE: sw
; NATIVE: ret
; MC-LABEL: <bound_fp16_dynamic>:
; MC: ttsfpmov.all l6, c9
; MC: sw
; MC: sw
; MC: ret
  %offset = zext i10 %index to i32
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 9)
  call void @llvm.riscv.tt.bound.sfpload(i32 6, i32 6, i32 %offset, i32 7, i32 1)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 %offset, i32 7, i32 1)
  ret void
}

;--- legacy.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @legacy_fp16() "tensix-executor"="trisc1" {
; LEGACY-LABEL: <legacy_fp16>:
; LEGACY: ttsfpload [[LREG:l[0-7]]], 0, 7, 1
; LEGACY: ttsfpstore [[LREG]], 2, 7, 1
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %old, i32 0, i32 7, i32 1)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %value, i32 2, i32 7, i32 1)
  ret void
}
