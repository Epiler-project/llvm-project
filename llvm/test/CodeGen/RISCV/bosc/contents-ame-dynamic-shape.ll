; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -regalloc=fast -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -filetype=obj -verify-machineinstrs %s -o %t.o0.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.o0.o | FileCheck %s --check-prefix=OBJ --implicit-check-not='<unknown>'
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -filetype=obj -verify-machineinstrs %s -o %t.o2.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.o2.o | FileCheck %s --check-prefix=OBJ --implicit-check-not='<unknown>'
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -regalloc=fast -filetype=obj -verify-machineinstrs %s -o %t.fast.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.fast.o | FileCheck %s --check-prefix=OBJ --implicit-check-not='<unknown>'

declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

; Independent runtime requests are bounded through real arithmetic. The
; accumulator's runtime stride is aligned and covers every configured N row.
; CHECK-LABEL: bounded_shape_and_stride:
; CHECK: msettilem
; CHECK: msettilen
; CHECK: msettilek
; CHECK: mlae8.m [[BA:tr[0-9]+]],
; CHECK: mlbe8.m [[BB:tr[0-9]+]],
; CHECK: mlce32.m [[BC:acc[0-9]+]], {{.*}}, [[BS:[a-z][a-z0-9]+]]
; CHECK: mqma.b.mm [[BC]], [[BA]], [[BB]]
; CHECK: msce32.m [[BC]], {{.*}}, [[BS]]
; ISEL-LABEL: name: bounded_shape_and_stride
; ISEL: BOSC_AME_SETM
; ISEL: BOSC_AME_SETN
; ISEL: BOSC_AME_SETK
; ISEL: %[[BA:[0-9]+]]:bosctrreg = BOSC_AME_LOADA
; ISEL: %[[BB:[0-9]+]]:bosctrreg = BOSC_AME_LOADB
; ISEL: %[[BC:[0-9]+]]:boscaccreg = BOSC_AME_LOADC {{.*}}, %[[BS:[0-9]+]], {{.*}} :: (volatile load unknown-size, align 1)
; ISEL: %[[BN:[0-9]+]]:boscaccreg = BOSC_AME_MMA %[[BC]], killed %[[BA]], killed %[[BB]],
; ISEL: BOSC_AME_STOREC killed %[[BN]], {{.*}}, %[[BS]], {{.*}} :: (volatile store unknown-size, align 1)
; OBJ-LABEL: <bounded_shape_and_stride>:
; OBJ: msettilem
; OBJ: msettilen
; OBJ: msettilek
; OBJ: mlae8.m [[BA:tr[0-9]+]],
; OBJ: mlbe8.m [[BB:tr[0-9]+]],
; OBJ: mlce32.m [[BC:acc[0-9]+]],
; OBJ: mqma.b.mm [[BC]], [[BA]], [[BB]]
; OBJ: msce32.m [[BC]],
define void @bounded_shape_and_stride(ptr %a, ptr %b, ptr %c, ptr %out, i64 %m_extent, i64 %n_extent, i64 %k_extent, i64 %padding) #0 {
  %m_low = and i64 %m_extent, 63
  %m_requested = add i64 %m_low, 1
  %n_low = and i64 %n_extent, 63
  %n_requested = add i64 %n_low, 1
  %k_low = and i64 %k_extent, 63
  %k_requested = add i64 %k_low, 1
  %pad = and i64 %padding, 252
  %stride = add i64 %pad, 256
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %m_requested)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %n_requested)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %k_requested)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 64, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}

; Each request may exceed 64. The hardware returns the actual clipped M/N/K;
; those exact SSA results drive row strides and the observable scalar outputs.
; Requested extents must not be substituted for returned actual dimensions.
; CHECK-LABEL: clipped_shape_and_actual_stride:
; CHECK: msettilem [[M:[a-z][a-z0-9]+]],
; CHECK: msettilen [[N:[a-z][a-z0-9]+]],
; CHECK: msettilek [[K:[a-z][a-z0-9]+]],
; CHECK-DAG: slli [[S:[a-z][a-z0-9]+]], [[N]], 2
; CHECK-DAG: mlae8.m [[A:tr[0-9]+]], {{.*}}, [[K]]
; CHECK: mlbe8.m [[B:tr[0-9]+]], {{.*}}, [[N]]
; CHECK: mlce32.m [[C:acc[0-9]+]], {{.*}}, [[S]]
; CHECK: mqma.b.mm [[C]], [[A]], [[B]]
; CHECK: msce32.m [[C]], {{.*}}, [[S]]
; CHECK: sd [[M]], 0(
; CHECK: sd [[N]], 8(
; CHECK: sd [[K]], 16(
; ISEL-LABEL: name: clipped_shape_and_actual_stride
; ISEL: %[[M:[0-9]+]]:gpr = BOSC_AME_SETM
; ISEL: %[[N:[0-9]+]]:gpr = BOSC_AME_SETN
; ISEL: %[[K:[0-9]+]]:gpr = BOSC_AME_SETK
; ISEL: %[[S:[0-9]+]]:gpr = SLLI %[[N]], 2
; ISEL: %[[A:[0-9]+]]:bosctrreg = BOSC_AME_LOADA {{.*}}, %[[K]], implicit $bosc_m, implicit $bosc_n, implicit $bosc_k
; ISEL: %[[B:[0-9]+]]:bosctrreg = BOSC_AME_LOADB {{.*}}, %[[N]], implicit $bosc_m, implicit $bosc_n, implicit $bosc_k
; ISEL: %[[C:[0-9]+]]:boscaccreg = BOSC_AME_LOADC {{.*}}, %[[S]],
; ISEL: %[[NEXT:[0-9]+]]:boscaccreg = BOSC_AME_MMA %[[C]], killed %[[A]], killed %[[B]],
; ISEL: BOSC_AME_STOREC killed %[[NEXT]], {{.*}}, %[[S]],
; ISEL: SD killed %[[M]], {{.*}}, 0,
; ISEL: SD %[[N]], {{.*}}, 8,
; ISEL: SD %[[K]], {{.*}}, 16,
; OBJ-LABEL: <clipped_shape_and_actual_stride>:
; OBJ: msettilem [[M:[a-z][a-z0-9]+]],
; OBJ: msettilen [[N:[a-z][a-z0-9]+]],
; OBJ: msettilek [[K:[a-z][a-z0-9]+]],
; OBJ-DAG: slli [[S:[a-z][a-z0-9]+]], [[N]], 0x2
; OBJ-DAG: mlae8.m [[A:tr[0-9]+]], {{.*}}, [[K]]
; OBJ: mlbe8.m [[B:tr[0-9]+]], {{.*}}, [[N]]
; OBJ: mlce32.m [[C:acc[0-9]+]], {{.*}}, [[S]]
; OBJ: mqma.b.mm [[C]], [[A]], [[B]]
; OBJ: msce32.m [[C]], {{.*}}, [[S]]
; OBJ: sd [[M]], 0x0(
; OBJ: sd [[N]], 0x8(
; OBJ: sd [[K]], 0x10(
define void @clipped_shape_and_actual_stride(ptr %a, ptr %b, ptr %c, ptr %out, ptr %actual, i64 %m_extent, i64 %n_extent, i64 %k_extent) #0 {
  %m_low = and i64 %m_extent, 127
  %m_requested = add i64 %m_low, 1
  %n_low = and i64 %n_extent, 127
  %n_requested = add i64 %n_low, 1
  %k_low = and i64 %k_extent, 127
  %k_requested = add i64 %k_low, 1
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %m_requested)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %n_requested)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %k_requested)
  %stride = shl i64 %n, 2
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 %k, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 %n, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  store volatile i64 %m, ptr %actual
  %actual_n = getelementptr i64, ptr %actual, i64 1
  store volatile i64 %n, ptr %actual_n
  %actual_k = getelementptr i64, ptr %actual, i64 2
  store volatile i64 %k, ptr %actual_k
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
