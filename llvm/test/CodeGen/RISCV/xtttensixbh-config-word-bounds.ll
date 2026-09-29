; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=VALID
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/static0.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/dynamic0.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC0
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/static1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/dynamic1.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC1
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/static2.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/dynamic2.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC2
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/static3.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/dynamic3.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC3
; STATIC: immediate operand 0 must be in [0, 223]
; DYNAMIC0: RMWCIB0 field cfgregaddr must be proven in [0, 223]
; DYNAMIC1: RMWCIB1 field cfgregaddr must be proven in [0, 223]
; DYNAMIC2: RMWCIB2 field cfgregaddr must be proven in [0, 223]
; DYNAMIC3: RMWCIB3 field cfgregaddr must be proven in [0, 223]
;
; The eight-bit instruction field addresses only 224 backend configuration
; words. 224..255 are not valid even though they fit the encoded field.
; The canonical machine descriptor must reject both immediate and port
; entrypoints; masking the data and mask fields must not reduce their range.

;--- valid.ll
declare void @llvm.riscv.tt.rmwcib0(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.rmwcib0.port(i32 immarg, i32, i32, i32)
declare void @llvm.riscv.tt.rmwcib1(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.rmwcib1.port(i32 immarg, i32, i32, i32)
declare void @llvm.riscv.tt.rmwcib2(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.rmwcib2.port(i32 immarg, i32, i32, i32)
declare void @llvm.riscv.tt.rmwcib3(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.rmwcib3.port(i32 immarg, i32, i32, i32)
define void @valid(i1 noundef %choose) "tensix-executor"="trisc1" {
; VALID-LABEL: name: valid
  %word = select i1 %choose, i32 222, i32 223
; VALID: TTRMWCIB0 223, 255, 255
  call void @llvm.riscv.tt.rmwcib0(i32 223, i32 255, i32 255)
; VALID: TTRMWCIB1 223, 255, 255
  call void @llvm.riscv.tt.rmwcib1(i32 223, i32 255, i32 255)
; VALID: TTRMWCIB2 223, 255, 255
  call void @llvm.riscv.tt.rmwcib2(i32 223, i32 255, i32 255)
; VALID: TTRMWCIB3 223, 255, 255
  call void @llvm.riscv.tt.rmwcib3(i32 223, i32 255, i32 255)
; VALID: PseudoTTRMWCIB0Port
  call void @llvm.riscv.tt.rmwcib0.port(i32 0, i32 %word, i32 255, i32 255)
; VALID: PseudoTTRMWCIB1Port
  call void @llvm.riscv.tt.rmwcib1.port(i32 0, i32 %word, i32 255, i32 255)
; VALID: PseudoTTRMWCIB2Port
  call void @llvm.riscv.tt.rmwcib2.port(i32 0, i32 %word, i32 255, i32 255)
; VALID: PseudoTTRMWCIB3Port
  call void @llvm.riscv.tt.rmwcib3.port(i32 0, i32 %word, i32 255, i32 255)
  ret void
}

;--- static0.ll
declare void @llvm.riscv.tt.rmwcib0(i32 immarg, i32 immarg, i32 immarg)
define void @static0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.rmwcib0(i32 224, i32 255, i32 255)
  ret void
}

;--- dynamic0.ll
declare void @llvm.riscv.tt.rmwcib0.port(i32 immarg, i32, i32, i32)
define void @dynamic0(i1 noundef %choose) "tensix-executor"="trisc1" {
  %word = select i1 %choose, i32 223, i32 224
  call void @llvm.riscv.tt.rmwcib0.port(i32 0, i32 %word, i32 255, i32 255)
  ret void
}

;--- static1.ll
declare void @llvm.riscv.tt.rmwcib1(i32 immarg, i32 immarg, i32 immarg)
define void @static1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.rmwcib1(i32 224, i32 255, i32 255)
  ret void
}

;--- dynamic1.ll
declare void @llvm.riscv.tt.rmwcib1.port(i32 immarg, i32, i32, i32)
define void @dynamic1(i1 noundef %choose) "tensix-executor"="trisc1" {
  %word = select i1 %choose, i32 223, i32 224
  call void @llvm.riscv.tt.rmwcib1.port(i32 0, i32 %word, i32 255, i32 255)
  ret void
}

;--- static2.ll
declare void @llvm.riscv.tt.rmwcib2(i32 immarg, i32 immarg, i32 immarg)
define void @static2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.rmwcib2(i32 224, i32 255, i32 255)
  ret void
}

;--- dynamic2.ll
declare void @llvm.riscv.tt.rmwcib2.port(i32 immarg, i32, i32, i32)
define void @dynamic2(i1 noundef %choose) "tensix-executor"="trisc1" {
  %word = select i1 %choose, i32 223, i32 224
  call void @llvm.riscv.tt.rmwcib2.port(i32 0, i32 %word, i32 255, i32 255)
  ret void
}

;--- static3.ll
declare void @llvm.riscv.tt.rmwcib3(i32 immarg, i32 immarg, i32 immarg)
define void @static3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.rmwcib3(i32 224, i32 255, i32 255)
  ret void
}

;--- dynamic3.ll
declare void @llvm.riscv.tt.rmwcib3.port(i32 immarg, i32, i32, i32)
define void @dynamic3(i1 noundef %choose) "tensix-executor"="trisc1" {
  %word = select i1 %choose, i32 223, i32 224
  call void @llvm.riscv.tt.rmwcib3.port(i32 0, i32 %word, i32 255, i32 255)
  ret void
}
