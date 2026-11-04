; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov

; A cold ordinary LReg can acquire its first real numerical definition from
; Dst. Full LOAD has no old numerical operand: the actual reaching ENCC and
; configuration reset establish all effective lanes, unblocked Dst reads and
; disabled destination-index capture. Its FP32 mode overwrites every bit.
; No read.constant, undef, poison, repair MOV or hidden old input supplies an
; initial value. The native instruction still retains CC/config/Dst effects.
;
; ENCC only establishes effective CC. CONFIGReset must execute with all eight
; columns active to clear ROW_MASK and BLOCK_SFPU_RD_FROM_DEST vertically.
; Both predecessor paths establish these facts independently in the CFG.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

; ISEL-LABEL: name: first_full_load
; ISEL: :sfpr = TTSFPLOADFull 0, 0, 3, implicit
; ISEL-NOT: IMPLICIT_DEF
; ISEL-NOT: TTSFPMOV
; OBJ-LABEL: <first_full_load>:
; OBJ: ttsfpencc 3, 10
; OBJ: ttsfpconfig.reset
; OBJ: ttsfpload [[FIRST:l[0-7]]], 0, 0, 3
; OBJ-NOT: ttsfpload
; OBJ: ttsfpstore [[FIRST]], 4, 0, 3
define void @first_full_load() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

; Disabling predication enables all lanes even when the raw flags are false.
; The earlier reset still owns ROW_MASK/read-block/index configuration.
; ISEL-LABEL: name: disabled_cc_still_loads_all
; ISEL: :sfpr = TTSFPLOADFull 8, 0, 3, implicit
; ISEL-NOT: IMPLICIT_DEF
; ISEL-NOT: TTSFPMOV
; OBJ-LABEL: <disabled_cc_still_loads_all>:
; OBJ: ttsfpencc 3, 10
; OBJ: ttsfpconfig.reset
; OBJ: ttsfpencc 0, 10
; OBJ: ttsfpload [[DISABLED:l[0-7]]], 8, 0, 3
; OBJ-NOT: ttsfpload
; OBJ: ttsfpstore [[DISABLED]], 12, 0, 3
define void @disabled_cc_still_loads_all() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.encc(i32 0, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 12, i32 0, i32 3)
  ret void
}

; ISEL-LABEL: name: both_predecessors_configure_full_load
; ISEL: :sfpr = TTSFPLOADFull 16, 0, 3, implicit
; ISEL-NOT: IMPLICIT_DEF
; ISEL-NOT: TTSFPMOV
; OBJ-LABEL: <both_predecessors_configure_full_load>:
; OBJ: ttsfpload [[JOIN:l[0-7]]], 16, 0, 3
; OBJ-NOT: ttsfpload
; OBJ: ttsfpstore [[JOIN]], 20, 0, 3
define void @both_predecessors_configure_full_load(i1 %choose) "tensix-executor"="trisc1" {
entry:
  br i1 %choose, label %left, label %right
left:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
right:
  call void @llvm.riscv.tt.contents.encc(i32 0, i32 2)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
join:
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 16, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 20, i32 0, i32 3)
  ret void
}

; A later partial LOAD still consumes the exact actual old definition. Neither
; full-word format nor the earlier initialization permits dropping that old
; operand once data-dependent lane predication or a halfword mode is used.
; ISEL-LABEL: name: partial_load_retains_defined_old
; ISEL: [[INITIAL:%[0-9]+]]:sfpr = TTSFPLOADFull 24, 0, 3, implicit
; ISEL: TTSFPSETCCEQ [[INITIAL]],
; ISEL: :sfpr = TTSFPLOAD [[INITIAL]], 28, 0, 14, implicit
; OBJ-LABEL: <partial_load_retains_defined_old>:
; OBJ: ttsfpload [[PARTIAL:l[0-7]]], 24, 0, 3
; OBJ: ttsfpsetcc.eq0 [[PARTIAL]]
; OBJ: ttsfpload [[PARTIAL]], 28, 0, 14
; OBJ: ttsfpencc 3, 10
; OBJ: ttsfpstore [[PARTIAL]], 32, 0, 3
define void @partial_load_retains_defined_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 24, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %initial, i32 6)
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %initial, i32 28, i32 0, i32 14)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %partial, i32 32, i32 0, i32 3)
  ret void
}
