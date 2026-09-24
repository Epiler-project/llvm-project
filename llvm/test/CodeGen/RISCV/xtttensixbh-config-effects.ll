; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s

declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @config_effects(i16 noundef %input) "tensix-executor"="trisc1" {
; CHECK-LABEL: name: config_effects
; CHECK: TTSETC16 2, 1
; CHECK-SAME: implicit-def {{(dead )?}}$tt_config
; CHECK-SAME: implicit-def {{(dead )?}}$tt_issue
; CHECK-SAME: implicit $tt_config
; CHECK-SAME: implicit $tt_issue
; CHECK: PseudoTTSETC16Port
; CHECK-SAME: implicit-def {{(dead )?}}$tt_config
; CHECK-SAME: implicit-def {{(dead )?}}$tt_issue
; CHECK-SAME: implicit $tt_config
; CHECK-SAME: implicit $tt_issue
  call void @llvm.riscv.tt.setc16(i32 2, i32 1)
  %value = zext i16 %input to i32
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 %value, i32 1)
  ret void
}
