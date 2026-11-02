; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; The Dst coordinate remains ordinary scalar SSA. Only numerical contents go
; through SFPR allocation; native dynamic issue exposes GPR word/address scratch
; separately, without turning the Dst coordinate into a memory pointer.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dynamic_dst(i32 noundef %offset) "tensix-executor"="trisc1" {
; CHECK-LABEL: dynamic_dst:
; CHECK: .word
; CHECK: sw
; CHECK: sw
; CHECK: ret
; ISEL-LABEL: name: dynamic_dst
; ISEL: :sfpr, {{.*}}:gpr, {{.*}}:gpr = PseudoTTSFPLOAD
; ISEL: :gpr, {{.*}}:gpr = PseudoTTSFPSTORE
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %bounded = and i32 %offset, 1023
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 %bounded, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 %bounded, i32 0, i32 3)
  ret void
}
