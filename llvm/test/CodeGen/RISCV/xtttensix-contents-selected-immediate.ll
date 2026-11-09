; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "

; A finite logical group is allocated as independent real SFPR values. Only
; the selected cell is changed. The one runtime word does not stage a vector
; selection, issue a MOV, duplicate a numerical operation, or duplicate loops.
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.3(<32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.5(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.6(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.7(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)

define void @selected_2(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_2
; POST: PseudoTTSFPContentsLoadImmediateSelected2
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_2>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 1
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %old0, <32 x i32> %old1, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_3(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_3
; POST: PseudoTTSFPContentsLoadImmediateSelected3
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_3>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 2
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.3(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_4(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_4
; POST: PseudoTTSFPContentsLoadImmediateSelected4
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_4>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 3
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.4(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_5(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_5
; POST: PseudoTTSFPContentsLoadImmediateSelected5
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_5>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 4
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.5(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_6(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_6
; POST: PseudoTTSFPContentsLoadImmediateSelected6
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_6>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 5
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.6(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_7(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_7
; POST: PseudoTTSFPContentsLoadImmediateSelected7
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_7>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old6 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 6
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.7(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5, <32 x i32> %old6, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_8(i32 noundef %selector) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_8
; POST: PseudoTTSFPContentsLoadImmediateSelected8
; POST-SAME: :: (volatile store (s32))
; OBJ-LABEL: <selected_8>:
; OBJ: lui [[MAP:[a-z0-9]+]], {{0x[0-9a-f]+}}
; OBJ-NEXT: addi [[MAP]], [[MAP]], {{-?(0x)?[0-9a-f]+}}
; OBJ-NEXT: slli [[FIELD:[a-z0-9]+]], {{[a-z0-9]+}}, {{(0x)?2}}
; OBJ-NEXT: srl [[FIELD]], [[MAP]], [[FIELD]]
; OBJ-NEXT: andi [[FIELD]], [[FIELD]], {{(0x)?7}}
; OBJ-NEXT: slli [[FIELD]], [[FIELD]], {{(0x14|20)}}
; OBJ-NEXT: lui [[BASE:[a-z0-9]+]], 0x71084
; OBJ-NEXT: addi [[BASE]], [[BASE]], {{(-128|-0x80)}}
; OBJ-NEXT: or [[BASE]], [[BASE]], [[FIELD]]
; OBJ-NEXT: lui [[PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[BASE]], {{(0x)?0}}([[PORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[PORT]])
; OBJ: ret
  %old0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old4 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old5 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old6 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %old7 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %choose = icmp eq i32 %selector, 0
  %slot = select i1 %choose, i32 0, i32 7
  %updated = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.8(<32 x i32> %old0, <32 x i32> %old1, <32 x i32> %old2, <32 x i32> %old3, <32 x i32> %old4, <32 x i32> %old5, <32 x i32> %old6, <32 x i32> %old7, i32 %slot, i32 16256, i32 8)
  ret void
}

define void @selected_loop(i32 noundef %count) "tensix-executor"="trisc1" {
; POST-LABEL: name: selected_loop
; POST: PseudoTTSFPContentsLoadImmediateSelected2
; POST-NOT: PseudoTTSFPContentsLoadImmediateSelected2
; POST: BNE
; OBJ-LABEL: <selected_loop>:
; OBJ: beqz
; OBJ: srl
; OBJ: lui [[LOOPPORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[LOOPPORT]])
; OBJ-NOT: sw {{[a-z0-9]+}}, {{(0x)?0}}([[LOOPPORT]])
; OBJ: bnez
; OBJ: ret
entry:
  %entry0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %entry1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %old0 = phi <32 x i32> [ %entry0, %entry ], [ %new0, %loop ]
  %old1 = phi <32 x i32> [ %entry1, %entry ], [ %new1, %loop ]
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  %slot = and i32 %remaining, 1
  %updated = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.immediate.selected.2(<32 x i32> %old0, <32 x i32> %old1, i32 %slot, i32 16256, i32 8)
  %new0 = extractvalue { <32 x i32>, <32 x i32> } %updated, 0
  %new1 = extractvalue { <32 x i32>, <32 x i32> } %updated, 1
  %next = add i32 %remaining, -1
  %again = icmp ne i32 %next, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
