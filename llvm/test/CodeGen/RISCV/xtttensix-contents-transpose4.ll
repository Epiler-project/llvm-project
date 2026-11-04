; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov --implicit-check-not=ttsfpload
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov --implicit-check-not=ttsfpload
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov --implicit-check-not=ttsfpload
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov --implicit-check-not=ttsfpload

; The four inputs are actual numerical values. L4..L7 are scratch clobbers,
; never additional old inputs or fabricated numerical results. Allocation
; sees those clobbers before the existing TRANSP word is selected.
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

; ISEL-LABEL: name: transpose_four_with_scratch
; ISEL: class: sfpr0
; ISEL: class: sfpr3
; ISEL: PseudoTTSFPContentsTranspose4 {{.*}}implicit-def {{(dead )?}}$tt_l4, implicit-def {{(dead )?}}$tt_l5, implicit-def {{(dead )?}}$tt_l6, implicit-def {{(dead )?}}$tt_l7, implicit-def {{(dead )?}}$tt_issue
; ISEL-NOT: PseudoTTSFPContentsTranspose8
; POST-LABEL: name: transpose_four_with_scratch
; POST: TTSFPTRANSP implicit-def {{(dead )?}}$tt_l0, implicit-def {{(dead )?}}$tt_l1, implicit-def {{(dead )?}}$tt_l2, implicit-def {{(dead )?}}$tt_l3, implicit-def {{(dead )?}}$tt_l4, implicit-def {{(dead )?}}$tt_l5, implicit-def {{(dead )?}}$tt_l6, implicit-def {{(dead )?}}$tt_l7, implicit-def {{(dead )?}}$tt_issue, implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3, implicit $tt_cc, implicit $tt_config, implicit $tt_issue{{$}}
; POST-NOT: PseudoTTSFPContentsTranspose
; OBJ-LABEL: <transpose_four_with_scratch>:
; OBJ: ttsfpmov.all l0, c9
; OBJ: ttsfpmov.all l1, c10
; OBJ: ttsfpmov.all l2, c15
; OBJ: ttsfpmov.all l3, c8
; OBJ-NOT: ttsfpmov.all l4,
; OBJ-NOT: ttsfpmov.all l5,
; OBJ-NOT: ttsfpmov.all l6,
; OBJ-NOT: ttsfpmov.all l7,
; OBJ: ttsfptransp
; OBJ-NOT: ttsfptransp
; OBJ: ttsfpstore l0,
; OBJ: ttsfpstore l1,
; OBJ: ttsfpstore l2,
; OBJ: ttsfpstore l3,
; OBJ-NOT: ttsfpstore
define void @transpose_four_with_scratch() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose4(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}
