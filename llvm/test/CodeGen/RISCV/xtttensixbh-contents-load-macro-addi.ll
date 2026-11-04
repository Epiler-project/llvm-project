; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; BH explicitly resets RWC.Sp and DEST_SP_BASE before ordinary FP32 seed,
; launch and observer access. Actual native setup, macro programming and tied
; L0..L3 allocation and a finite queue drain. No ordinary ADD substitutes for
; macro arithmetic, and the scalar loop is preserved rather than replayed.
declare void @llvm.riscv.tt.contents.macro.address.reset()
declare void @llvm.riscv.tt.contents.macro.configure.addi(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @bh_macro_addi_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <bh_macro_addi_loop>:
; CHECK: ttsetc16 0, 0
; CHECK-NEXT: {{.*}}ttnop
; CHECK-NEXT: {{.*}}ttnop
; CHECK-NEXT: {{.*}}ttnop
; CHECK: ttsetc16 0, 1
; CHECK: ttrmwcib0 6, 0, 255
; CHECK: ttrmwcib1 6, 0, 255
; CHECK: ttrmwcib0 7, 0, 255
; CHECK: ttrmwcib1 7, 0, 255
; CHECK: ttsetc16 0, 28
; CHECK: ttsetrwc 20, 0, 0, 0, 0, 0
; CHECK: ttsfpconfig.macro.template 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK: ttsfpconfig.macro.sequence 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK: ttsfpconfig.macro.misc 3
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK: ttsfpstore [[VALUE:l[0-3]]], 16, 0, 3
; CHECK: ttsfploadmacro.fp32.even [[VALUE]], 8, 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpload [[VALUE]], 16, 0, 3
; CHECK: ttsfpstore [[VALUE]], 2, 0, 3
; ISEL-LABEL: name: bh_macro_addi_loop
; ISEL: PseudoTTSFPMacroAddressReset
; ISEL: PseudoTTSFPMacroConfigureADDI 0, 16256
; ISEL: :sfprlow = PseudoTTSFPContentsLoadMacroADDI
entry:
  call void @llvm.riscv.tt.contents.macro.address.reset()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.macro.configure.addi(i32 0, i32 16256)
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %seed, <32 x i32> %seed, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %initial, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %initial, i32 16, i32 0, i32 3)
  %empty = icmp eq i32 %limit, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%initial, %entry], [%sum, %loop]
  %sum = call <32 x i32> @llvm.riscv.tt.contents.load.macro.addi(<32 x i32> %value, i32 16, i32 0)
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %last = phi <32 x i32> [%initial, %entry], [%sum, %loop]
  %observed = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %last, i32 16, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %observed, i32 2, i32 0, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
