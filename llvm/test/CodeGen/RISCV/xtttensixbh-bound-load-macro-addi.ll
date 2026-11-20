; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=BUNDLE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARCH
; RUN: sed 's/"trisc1"/"trisc0"/' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROLE
; RUN: sed 's/load.macro.addi(i32 3, i32 16/load.macro.addi(i32 4, i32 16/' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=LREG
; RUN: sed 's/load.macro.addi(i32 3, i32 16/load.macro.addi(i32 3, i32 17/' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: sed '/call void @llvm.tt.bh.sfpu.macro.address.reset()/d' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDRESS
; RUN: sed '/call void @llvm.tt.bh.sfpu.macro.configure.addi/d' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: sed 's/@llvm.tt.bh.sfpu.config.reset/@llvm.riscv.tt.contents.config.reset/g' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIX

; OBJ-LABEL: <bound_macro_loop>:
; OBJ: ttsetc16 0, 20
; OBJ: ttsfpconfig.macro.template 2
; OBJ: ttsfpconfig.macro.sequence 2
; OBJ: ttsfpconfig.macro.misc 3
; OBJ: ttsfpstore l3, 16, 0, 3
; OBJ: ttsfploadmacro.fp32.even l3, 8, 2
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ: bne{{z?}} {{.*}},
; OBJ: ttsfpload l3, 16, 0, 3
; OBJ: ttsfpstore l3, 2, 0, 3

; ISEL-LABEL: name: bound_macro_loop
; ISEL-NOT: :sfpr
; ISEL: $tt_l3 = PseudoTTSFPLoadMacroADDI $tt_l3, 8, 2
; ISEL-NOT: :sfpr
; BUNDLE-LABEL: name: bound_macro_loop
; BUNDLE: BUNDLE {{.*}} :: (volatile load store (s32)) {
; BUNDLE-NEXT: {{.*}}$tt_l3 = TTSFPLOADMACROFP32Even {{.*}}, 8, 2,
; BUNDLE-NEXT: TTSFPNOP
; BUNDLE-NEXT: TTSFPNOP
; BUNDLE-NEXT: TTSFPNOP
; BUNDLE-NEXT: TTSFPNOP
; BUNDLE-NEXT: }
; ARCH: SFPU intrinsic architecture does not match target feature
; ROLE: SFPLOADMACRO requires the TRISC1 executor
; LREG: SFPLOADMACRO requires physical LReg[0..3]
; OFFSET: SFPLOADMACRO FP32 offset must be an even constant in [0, 1022]
; ADDRESS: SFPLOADMACRO requires actual zero Dst address setup on every incoming path
; CONFIG: SFPLOADMACRO requires its actual ADDI template, sequence and FP32 misc configuration
; MIX: bound SFPU and contents SSA cannot mix in one function

declare void @llvm.tt.bh.sfpu.macro.address.reset()
declare void @llvm.riscv.tt.setibrwc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.macro.configure.addi(i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.load.macro.addi(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.config.reset()
declare void @llvm.tt.bh.sfpu.mov.all(i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.tt.bh.sfpu.load(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)

define void @bound_macro_loop(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.tt.bh.sfpu.macro.address.reset()
  call void @llvm.riscv.tt.setibrwc(i32 0, i32 0, i32 0)
  call void @llvm.tt.bh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.bh.sfpu.config.reset()
  call void @llvm.tt.bh.sfpu.macro.configure.addi(i32 2, i32 16256)
  call void @llvm.tt.bh.sfpu.mov.all(i32 3, i32 9)
  call void @llvm.tt.bh.sfpu.store(i32 3, i32 16, i32 0, i32 3)
  %empty = icmp eq i32 %limit, 0
  br i1 %empty, label %exit, label %loop

loop:
  %index = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.tt.bh.sfpu.load.macro.addi(i32 3, i32 16, i32 2)
  %next = add nuw i32 %index, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  call void @llvm.tt.bh.sfpu.load(i32 3, i32 3, i32 16, i32 0, i32 3)
  call void @llvm.tt.bh.sfpu.store(i32 3, i32 2, i32 0, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
