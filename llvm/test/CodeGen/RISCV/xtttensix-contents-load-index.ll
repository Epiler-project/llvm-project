; RUN: split-file %s %t
; RUN: cat %t/valid.ll %t/matrix.ll > %t/bh-valid.ll
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/bh-valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=BH,BHONLY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wh.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/bh-valid.ll -o - | FileCheck %s --check-prefixes=POST,BHPOST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/bh-valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=BH,BHONLY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wh.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/bh-valid.ll -o - | FileCheck %s --check-prefixes=POST,BHPOST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/bh-valid.ll -o - | FileCheck %s --check-prefixes=ISEL,BHISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/format.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/addrmod.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDRMOD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/format.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/addrmod.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDRMOD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/valid.ll -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/valid.ll -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL

; A LOAD can update the primary value and its L4 destination-index companion.
; Both real old contents and both results must survive allocation; disabled
; capture or blocked/inactive lanes preserve the old index. Formats 14/15
; additionally preserve the primary value's other halfword. No hidden SFPU
; copy, scratch or claimed LaneConfig state supplies either old contents.
; The receiver allocates a native primary/+4 pair; this source selects its
; static Dst offset without naming a physical register.
; Both targets exercise configured capture and alternate ROW_MASK bits. WH
; additionally checks capture disabled with its maximum native AddrMod.
; Dynamic Dst coordinates use the same coupled allocation representation.
; OFFSET: contents paired LOAD Dst offset must be in [0, 1023]
; FORMAT: SFPLOADIndex mode must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
; ADDRMOD: Wormhole SFPLOADIndex operand 5 must be in [0, 3]

; BH-LABEL: <load_index_capture>:
; BH: ttsfpload l0, 4, 0, 3
; BH: ttsfpstore l0,
; BH: ttsfpstore l4,
; ISEL-LABEL: name: load_index_capture
; ISEL: class: sfprindexpair
; ISEL: :sfprindexpair = PseudoTTSFPContentsLoadIndex %{{[0-9]+}}, 4, 0, 3,
; POST-LABEL: name: load_index_capture
; POST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4 = TTSFPLOADIndex {{(killed )?}}{{(renamable )?}}$tt_l0, {{(killed )?}}{{(renamable )?}}$tt_l4, 4, 0, 3,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l0,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l4,

; BH-LABEL: <load_index_masked_low>:
; BH: ttsfpload l0, 4, 0, 14
; BH: ttsfpstore l0,
; BH: ttsfpstore l4,
; ISEL-LABEL: name: load_index_masked_low
; ISEL: class: sfprindexpair
; ISEL: :sfprindexpair = PseudoTTSFPContentsLoadIndex %{{[0-9]+}}, 4, 0, 14,
; POST-LABEL: name: load_index_masked_low
; POST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4 = TTSFPLOADIndex {{(killed )?}}{{(renamable )?}}$tt_l0, {{(killed )?}}{{(renamable )?}}$tt_l4, 4, 0, 14,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l0,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l4,

; BH-LABEL: <load_index_masked_high>:
; BH: ttsfpload l0, 4, 0, 15
; BH: ttsfpstore l0,
; BH: ttsfpstore l4,
; ISEL-LABEL: name: load_index_masked_high
; ISEL: class: sfprindexpair
; ISEL: :sfprindexpair = PseudoTTSFPContentsLoadIndex %{{[0-9]+}}, 4, 0, 15,
; POST-LABEL: name: load_index_masked_high
; POST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4 = TTSFPLOADIndex {{(killed )?}}{{(renamable )?}}$tt_l0, {{(killed )?}}{{(renamable )?}}$tt_l4, 4, 0, 15,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l0,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l4,

; BH-LABEL: <load_index_disabled>:
; BH: ttsfpload l0, 4, 0, 3
; BH: ttsfpstore l0,
; BH: ttsfpstore l4,
; ISEL-LABEL: name: load_index_disabled
; ISEL: class: sfprindexpair
; ISEL: :sfprindexpair = PseudoTTSFPContentsLoadIndex %{{[0-9]+}}, 4, 0, 3,
; POST-LABEL: name: load_index_disabled
; POST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4 = TTSFPLOADIndex {{(killed )?}}{{(renamable )?}}$tt_l0, {{(killed )?}}{{(renamable )?}}$tt_l4, 4, 0, 3,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l0,
; POST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l4,

; BHONLY-LABEL: <load_index_matrix_hazard>:
; BHONLY: ttsfpload l0, 4, 0, 3
; BHONLY: ttsfpstore l0,
; BHONLY: ttsfpstore l4,
; BHISEL-LABEL: name: load_index_matrix_hazard
; BHISEL: class: sfprindexpair
; BHISEL: :sfprindexpair = PseudoTTSFPContentsLoadIndex %{{[0-9]+}}, 4, 0, 3,
; BHPOST-LABEL: name: load_index_matrix_hazard
; BHPOST: TTMVMUL
; BHPOST-NEXT: TTSFPNOP
; BHPOST-NEXT: TTSFPNOP
; BHPOST-NEXT: TTSFPNOP
; BHPOST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4 = TTSFPLOADIndex {{(killed )?}}{{(renamable )?}}$tt_l0, {{(killed )?}}{{(renamable )?}}$tt_l4, 4, 0, 3,
; BHPOST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l0,
; BHPOST: TTSFPSTORE {{(killed )?}}{{(renamable )?}}$tt_l4,

; WH-LABEL: <load_index_wh_addr3>:
; WH: ttsfpload l0, 4, 3, 3
; WH: ttsfpstore l0,
; WH: ttsfpstore l4,

;--- valid.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

declare void @llvm.riscv.tt.mvmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @load_index_capture() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

define void @load_index_masked_low() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 20492, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %oldvalue = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %config, i32 4660, i32 8)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %oldvalue, <32 x i32> %index, i32 4, i32 0, i32 14)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

define void @load_index_masked_high() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 40972, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %oldvalue = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %config, i32 4660, i32 8)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %oldvalue, <32 x i32> %index, i32 4, i32 0, i32 15)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

define void @load_index_disabled() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 0, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}


;--- wh.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define void @load_index_wh_addr3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 0, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 3, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

;--- offset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define void @invalid_offset() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 0, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 1024, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

;--- addrmod.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define void @invalid_addrmod() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 0, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 4, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

;--- format.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg, i32 immarg)

define void @invalid_format() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 0, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 0, i32 7)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

;--- matrix.ll
define void @load_index_matrix_hazard() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 4, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}
