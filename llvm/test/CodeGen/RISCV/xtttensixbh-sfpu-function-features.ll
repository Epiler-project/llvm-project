; The module/default target has no Tensix feature. Register allocation must
; follow each function's subtarget while preserving ordinary scalar code.
; Gating the entire SFPU allocation pipeline on the default target is invalid.
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

; MIR-LABEL: name: ordinary
; MIR: LW
; MIR: ADD
; MIR: SW
; MIR: PseudoRET
define i32 @ordinary(ptr %p, i32 %increment) {
  %old = load volatile i32, ptr %p, align 4
  %value = add i32 %old, %increment
  store volatile i32 %value, ptr %p, align 4
  ret i32 %value
}

declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)

; MIR-LABEL: name: tensix_loop
; MIR: $tt_l{{[0-7]}} = TTSFPLOADI
; MIR: $tt_l{{[0-7]}} = TTSFPIADD
; MIR: BNE
; MIR: $tt_l3 = TTSFPMOVAll
; MIR: PseudoRET
define void @tensix_loop(i32 %limit) "target-features"="+xtttensixbh" "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> %zero, i32 1, i32 2)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%zero, %entry], [%sum, %loop]
  %sum = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %value, <32 x i32> %one, i32 4)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %sum)
  ret void
}
