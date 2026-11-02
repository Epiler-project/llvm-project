; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; Hardware constants are actual fixed reads, not writable allocation capacity.
; In particular C15 is initialized even though its lanes contain different bits.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @immutable_roles() "tensix-executor"="trisc1" {
; CHECK-LABEL: <immutable_roles>:
; CHECK: ttsfpmov.all [[V0:l[0-7]]], c8
; CHECK: ttsfpstore [[V0]], 0, 0, 4
; CHECK: ttsfpmov.all [[V1:l[0-7]]], c9
; CHECK: ttsfpstore [[V1]], 2, 0, 4
; CHECK: ttsfpmov.all [[V2:l[0-7]]], c10
; CHECK: ttsfpstore [[V2]], 4, 0, 4
; CHECK: ttsfpmov.all [[V3:l[0-7]]], c15
; CHECK: ttsfpstore [[V3]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: immutable_roles
; ISEL: :sfpr = TTSFPMOVAll $tt_c8
; ISEL: :sfpr = TTSFPMOVAll $tt_c9
; ISEL: :sfpr = TTSFPMOVAll $tt_c10
; ISEL: :sfpr = TTSFPMOVAll $tt_c15
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v0, i32 0, i32 0, i32 4)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 2, i32 0, i32 4)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 4, i32 0, i32 4)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 6, i32 0, i32 4)
  ret void
}

; Each half-load preserves the other half of the preceding numerical value.
; Removing the tied old input would turn this into two unrelated definitions.
define void @partial_dst_contents() "tensix-executor"="trisc1" {
; CHECK-LABEL: <partial_dst_contents>:
; CHECK: ttsfpmov.all [[HALVES:l[0-7]]], c10
; CHECK: ttsfpload [[HALVES]], 0, 0, 14
; CHECK: ttsfpload [[HALVES]], 2, 0, 15
; CHECK: ttsfpstore [[HALVES]], 4, 0, 4
; CHECK: ret
; ISEL-LABEL: name: partial_dst_contents
; ISEL: %[[SEED:[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; ISEL: %[[LOW:[0-9]+]]:sfpr = TTSFPLOAD %[[SEED]], 0, 0, 14,
; ISEL: %[[BOTH:[0-9]+]]:sfpr = TTSFPLOAD %[[LOW]], 2, 0, 15,
; ISEL: TTSFPSTORE {{(killed )?}}%[[BOTH]], 4, 0, 4,
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %low = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 14)
  %both = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %low, i32 2, i32 0, i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %both, i32 4, i32 0, i32 4)
  ret void
}

; Eight simultaneously live values fit the ordinary L0-L7 bank. Together with
; the nine-value negative test this prevents fixed/staging roles from becoming
; extra capacity, and prevents an unnecessarily smaller ordinary register bank.
define void @eight_live_values() "tensix-executor"="trisc1" {
; CHECK-LABEL: <eight_live_values>:
; CHECK-COUNT-8: ttsfpmov.all {{l[0-7]}}, c9
; CHECK-COUNT-8: ttsfpstore {{l[0-7]}},
; CHECK: ret
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 6, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v4, i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v5, i32 10, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v6, i32 12, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v7, i32 14, i32 0, i32 3)
  ret void
}
