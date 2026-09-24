; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL

declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
declare void @llvm.riscv.tt.dependent.use(i32)

define void @direct() "tensix-executor"="trisc1" {
; CHECK-LABEL: direct:
; CHECK: .word 0xc804000a
; CHECK-NOT: call
; CHECK: ret
; O0-LABEL: direct:
; O0: .word 0xc804000a
; ISEL-LABEL: name: direct
; ISEL: TTSETC16 2, 1
  call void @llvm.riscv.tt.setc16(i32 2, i32 1)
  ret void
}

define void @local_port(i16 noundef zeroext %input) "tensix-executor"="trisc2" {
; CHECK-LABEL: local_port:
; CHECK: lui [[WORD:[a-z0-9]+]], 729088
; CHECK: or [[WORD]], [[WORD]], {{[a-z0-9]+}}
; CHECK: lui [[PORT:[a-z0-9]+]], 1048128
; CHECK: sw [[WORD]], 0([[PORT]])
; CHECK: ret
; O0-LABEL: local_port:
; O0: sw
; ISEL-LABEL: name: local_port
; ISEL: early-clobber
; ISEL-SAME: PseudoTTSETC16Port
  %value = zext i16 %input to i32
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 %value, i32 1)
  ret void
}

define void @brisc_remote_ports(i16 noundef zeroext %input) "tensix-executor"="brisc" {
; CHECK-LABEL: brisc_remote_ports:
; CHECK: lui [[P1:[a-z0-9]+]], 1048144
; CHECK: sw {{[a-z0-9]+}}, 0([[P1]])
; CHECK: lui [[P2:[a-z0-9]+]], 1048160
; CHECK: sw {{[a-z0-9]+}}, 0([[P2]])
; CHECK: ret
  %value = zext i16 %input to i32
  call void @llvm.riscv.tt.setc16.port(i32 1, i32 %value, i32 19)
  call void @llvm.riscv.tt.setc16.port(i32 2, i32 %value, i32 35)
  ret void
}

define void @dependent_read(ptr %address, ptr %output) "tensix-executor"="ncrisc" {
; CHECK-LABEL: dependent_read:
; CHECK: lw [[READ:[a-z0-9]+]], 0(a0)
; CHECK: and zero, zero, [[READ]]
; CHECK: sw
; CHECK: ret
  %value = load volatile i32, ptr %address
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  store volatile i32 1, ptr %output
  ret void
}

define void @retained_loop(i16 %limit) "tensix-executor"="trisc0" {
; CHECK-LABEL: retained_loop:
; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: sw
; CHECK: bne {{.*}}, .LBB{{[0-9]+}}_1
; CHECK: ret
entry:
  br label %loop
loop:
  %i = phi i16 [0, %entry], [%next, %loop]
  %value = zext i16 %i to i32
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 %value, i32 19)
  %next = add i16 %i, 1
  %done = icmp eq i16 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
