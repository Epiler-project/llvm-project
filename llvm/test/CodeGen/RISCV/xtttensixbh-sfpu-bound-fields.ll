; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/reserved.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RESERVED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/transpose.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIXED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vector-field.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vector-mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/immediate.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=IMM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/stochrnd.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESCALE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset-range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 %t/feature.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/attribute.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ATTRIBUTE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/constant.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/carrier.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CARRIER
; RESERVED: bound SFPU logical field operand 1 must be in [0, 0]
; CONFIG: SFPCONFIG mode 8 mask must select even lane bits
; FIXED: bound SFPU fixed register group does not match the instruction
; TIE: bound SFPU destructive tie requires identical physical registers
; MODE: bound SFPU unsupported mode
; IMM: Tensix immediate operand must fit in 16 unsigned bits
; DESCALE: Tensix floating conversion requires zero descale; integer descale must fit five bits
; OFFSET: bound Dst offset must be proven in [0, 1023]
; FEATURE: Tensix intrinsic requires +xtttensixbh
; ATTRIBUTE: Tensix intrinsic requires a tensix-executor function attribute
; CONSTANT: immarg operand has non-immediate parameter
; CARRIER: SFPU carrier constants require explicit target initialization

;--- reserved.ll
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
define void @reserved() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 0, i32 1, i32 2)
  ret void
}
;--- config.ll
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @config() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 2, i32 8)
  ret void
}
;--- transpose.ll
declare void @llvm.riscv.tt.bound.sfptransp(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @transpose() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfptransp(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 0, i32 0, i32 1, i32 2, i32 3)
  ret void
}
;--- vector-field.ll
declare void @llvm.riscv.tt.bound.sfpsetexp.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @vector_field() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpsetexp.v(i32 2, i32 2, i32 0, i32 3, i32 2)
  ret void
}
;--- vector-mode.ll
declare void @llvm.riscv.tt.bound.sfpshft.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @vector_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpshft.v(i32 2, i32 2, i32 0, i32 1)
  ret void
}
;--- immediate.ll
declare void @llvm.riscv.tt.bound.sfploadi(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @immediate() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfploadi(i32 0, i32 0, i32 65536, i32 2)
  ret void
}
;--- stochrnd.ll
declare void @llvm.riscv.tt.bound.sfpstochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @stochrnd() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpstochrnd.i(i32 0, i32 0, i32 1, i32 32, i32 4, i32 0)
  ret void
}
;--- offset-range.ll
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @offset_range(i32 noundef %address) "tensix-executor"="trisc1" {
  %offset = and i32 %address, 2047
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %offset, i32 0, i32 4)
  ret void
}
;--- feature.ll
declare void @llvm.riscv.tt.bound.sfpnop()
define void @feature() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpnop()
  ret void
}
;--- attribute.ll
declare void @llvm.riscv.tt.bound.sfpnop()
define void @attribute() {
  call void @llvm.riscv.tt.bound.sfpnop()
  ret void
}
;--- constant.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
define void @constant(i32 noundef %number) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 %number, i32 9)
  ret void
}
;--- carrier.ll
declare void @llvm.riscv.tt.bound.sfpnop()
define void @carrier(i1 %again, ptr %sink) "tensix-executor"="trisc1" {
entry:
  br label %loop
loop:
  %unbound = phi <32 x i32> [zeroinitializer, %entry], [%unbound, %loop]
  store volatile <32 x i32> %unbound, ptr %sink
  call void @llvm.riscv.tt.bound.sfpnop()
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
