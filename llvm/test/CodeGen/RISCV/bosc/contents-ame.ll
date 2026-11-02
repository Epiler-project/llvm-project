; RUN: llc -mtriple=riscv64 -mattr=+xboscame -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -enable-new-pm -O0 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -enable-new-pm -O2 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; NEWPM: requires the SelectionDAG legacy codegen pipeline

; Dedicated opaque contents retain the matrix register-file role across PHIs.
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @matrix_contents(ptr %a, ptr %b, ptr %c, ptr %out) #0 {
; CHECK-LABEL: matrix_contents:
; CHECK: mlae8.m [[A:tr[0-9]+]],
; CHECK: mlbe8.m [[B:tr[0-9]+]],
; CHECK: mlce32.m [[C:acc[0-9]+]],
; CHECK: mqma.b.mm [[C]], [[A]], [[B]]
; CHECK: msce32.m [[C]],
; ISEL-LABEL: name: matrix_contents
; ISEL: class: bosctrreg
; ISEL: class: boscaccreg
; ISEL: BOSC_AME_LOADA {{.*}} :: (volatile load unknown-size, align 1)
; ISEL: BOSC_AME_LOADB {{.*}} :: (volatile load unknown-size, align 1)
; ISEL: BOSC_AME_LOADC {{.*}} :: (volatile load unknown-size, align 1)
; ISEL: BOSC_AME_MMA
; ISEL: BOSC_AME_STOREC {{.*}} :: (volatile store unknown-size, align 1)
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 16, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

define void @loop_contents(ptr %a, ptr %b, ptr %c, ptr %out, i64 %count) #0 {
; CHECK-LABEL: loop_contents:
; CHECK: mlce32.m [[ACC:acc[0-9]+]],
; CHECK: [[LOOP:.LBB[0-9_]+]]:
; CHECK: mqma.b.mm [[ACC]],
; CHECK: b{{[a-z]+}} {{.*}}[[LOOP]]
; CHECK: msce32.m [[ACC]],
; ISEL-LABEL: name: loop_contents
; ISEL: PHI
; ISEL: BOSC_AME_MMA
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %seedk = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %initial = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %n, i64 %seedk)
  br label %loop
loop:
  %i = phi i64 [ 0, %0 ], [ %inc, %loop ]
  %acc = phi target("riscv.bosc.ame.acc") [ %initial, %0 ], [ %next, %loop ]
  %bounded = and i64 %i, 63
  %requested = add i64 %bounded, 1
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %requested)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 16, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %acc, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  %inc = add i64 %i, 1
  %again = icmp ult i64 %inc, %count
  br i1 %again, label %loop, label %done
done:
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

attributes #0 = { "trex-bosc-contents-ame"="1" }
