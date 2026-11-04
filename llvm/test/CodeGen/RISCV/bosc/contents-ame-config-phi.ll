; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/positive.ll -o %t/o0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/positive.ll -o %t/o2.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs %t/positive.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs %t/positive.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -stop-after=finalize-isel %t/positive.ll -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -stop-after=finalize-isel %t/positive.ll -o - | FileCheck %s --check-prefix=MIR
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/wrong-role.ll -o %t/wrong-role-O0.o 2>&1 | FileCheck %s --check-prefix=ORIGIN
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/wrong-role.ll -o %t/wrong-role-O2.o 2>&1 | FileCheck %s --check-prefix=ORIGIN
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/old-shape.ll -o %t/old-shape-O0.o 2>&1 | FileCheck %s --check-prefix=ORIGIN
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/old-shape.ll -o %t/old-shape-O2.o 2>&1 | FileCheck %s --check-prefix=ORIGIN
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/wrong-row.ll -o %t/wrong-row-O0.o 2>&1 | FileCheck %s --check-prefix=STRIDE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/wrong-row.ll -o %t/wrong-row-O2.o 2>&1 | FileCheck %s --check-prefix=STRIDE
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/stale-config.ll -o %t/stale-config-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/stale-config.ll -o %t/stale-config-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/forged-config.ll -o %t/forged-config-O0.o 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/forged-config.ll -o %t/forged-config-O2.o 2>&1 | FileCheck %s --check-prefix=CONFIG

; Configuration and contents must select the same predecessor, retaining role
; and row coverage. Scalar equivalence or a matching set of PHI operands alone
; does not establish that association. No allocator or producer API changes.
;
; ORIGIN: AME contents have incompatible role or configuration
; STRIDE: AME contents stride does not cover its configured row
; CONFIG: AME contents require the actual reaching M/N/K configuration
;
; ASM-LABEL: ame_branch_config:
; ASM: b{{[a-z]+}}
; ASM: mqma.b.mm
; ASM: msce32.m
; ASM-LABEL: branch_geometry:
; ASM: b{{[a-z]+}}
; ASM: mqma.b.mm
; ASM: msce32.m
; ASM-LABEL: join_loads:
; ASM: b{{[a-z]+}}
; ASM: mlae8.m
; ASM: mlbe8.m
; ASM: mlce32.m
; ASM: mqma.b.mm
; ASM: msce32.m
;
; MIR-LABEL: name: ame_branch_config
; MIR: class: bosctrreg
; MIR: class: boscaccreg
; MIR: PHI
; MIR: BOSC_AME_MMA
; MIR-LABEL: name: branch_geometry
; MIR: class: bosctrreg
; MIR: class: boscaccreg
; MIR: PHI
; MIR: BOSC_AME_MMA
; MIR-LABEL: name: join_loads
; MIR: class: bosctrreg
; MIR: class: boscaccreg
; MIR: BOSC_AME_LOADA
; MIR: BOSC_AME_LOADB
; MIR: BOSC_AME_LOADC
; MIR: BOSC_AME_MMA

;--- positive.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @ame_branch_config(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %n, i64 %k)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %n, i64 %k)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %n, i64 %k)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %mr, i64 %n, i64 %k)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %mr, i64 %n, i64 %k)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %mr, i64 %n, i64 %k)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %ar, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 28, i64 %m, i64 %n, i64 %k)
  ret void
}

define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %ar, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}

define void @join_loads(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %astride = phi i64 [ 4, %left ], [ 6, %right ]
  %bstride = phi i64 [ 7, %left ], [ 5, %right ]
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 %astride, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 %bstride, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 %stride, i64 %m, i64 %n, i64 %k)
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}

attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- wrong-role.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %br, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- old-shape.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  %old_m = call i64 @llvm.riscv.bosc.ame.setm(i64 4)
  %old_n = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %old_k = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %old_a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %old_m, i64 %old_n, i64 %old_k)
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %old_a, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- wrong-row.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 16, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %ar, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- stale-config.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  %later_m = call i64 @llvm.riscv.bosc.ame.setm(i64 1)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ %mr, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %ar, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- forged-config.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @branch_geometry(ptr %a, ptr %b, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  br i1 %choose, label %left, label %right
left:
  %ml = call i64 @llvm.riscv.bosc.ame.setm(i64 3)
  %nl = call i64 @llvm.riscv.bosc.ame.setn(i64 7)
  %kl = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  %al = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 4, i64 %ml, i64 %nl, i64 %kl)
  %bl = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 7, i64 %ml, i64 %nl, i64 %kl)
  %cl = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 28, i64 %ml, i64 %nl, i64 %kl)
  br label %join
right:
  %mr = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %nr = call i64 @llvm.riscv.bosc.ame.setn(i64 5)
  %kr = call i64 @llvm.riscv.bosc.ame.setk(i64 6)
  %ar = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 6, i64 %mr, i64 %nr, i64 %kr)
  %br = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 5, i64 %mr, i64 %nr, i64 %kr)
  %cr = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 20, i64 %mr, i64 %nr, i64 %kr)
  br label %join
join:
  %m = phi i64 [ %ml, %left ], [ 2, %right ]
  %n = phi i64 [ %nl, %left ], [ %nr, %right ]
  %k = phi i64 [ %kl, %left ], [ %kr, %right ]
  %stride = phi i64 [ 28, %left ], [ 20, %right ]
  %av = phi target("riscv.bosc.ame.tr") [ %al, %left ], [ %ar, %right ]
  %bv = phi target("riscv.bosc.ame.tr") [ %bl, %left ], [ %br, %right ]
  %cv = phi target("riscv.bosc.ame.acc") [ %cl, %left ], [ %cr, %right ]
  %r = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %r, ptr %out, i64 %stride, i64 %m, i64 %n, i64 %k)
  ret void
}
attributes #0 = { "trex-bosc-contents-ame"="1" }
