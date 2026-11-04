; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPMOV
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,BH --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,WH --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,BH --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,WH --implicit-check-not=ttsfpmov
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "

; A runtime Dst coordinate changes only instruction-word construction. The
; full result still has no numerical old operand, and both GPR scratches are
; explicit, early-clobber and independent of the actual scalar coordinate.
; AddrMod=3 retains the actual WH and BH field encodings. The second function
; retains its numerical/scalar PHIs and zero-trip edge without source unroll.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dynamic_first_full_load(i32 noundef %offset) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_first_full_load
; ISEL: [[OFFSET:%[0-9]+]]:gpr = ANDI
; ISEL: [[VALUE:%[0-9]+]]:sfpr, early-clobber {{%[0-9]+}}:gpr, early-clobber {{%[0-9]+}}:gpr = PseudoTTSFPLOADFull {{(killed )?}}[[OFFSET]], 3, 3, implicit
; ISEL: TTSFPSTORE {{(killed )?}}[[VALUE]], 4, 0, 3,
; POST-LABEL: name: dynamic_first_full_load
; POST: {{(renamable )?}}$tt_l0, {{(dead )?}}early-clobber {{(renamable )?}}${{[a-z0-9]+}}, {{(dead )?}}early-clobber {{(renamable )?}}${{[a-z0-9]+}} = PseudoTTSFPLOADFull {{(killed )?(renamable )?}}${{[a-z0-9]+}}, 3, 3,
; POST-SAME: :: (volatile load store (s32))
; OBJ-LABEL: <dynamic_first_full_load>:
; BH: lui [[WORD:[a-z0-9]+]], 0x70036
; WH: lui [[WORD:[a-z0-9]+]], 0x7003c
; OBJ-NEXT: or [[WORD]], [[WORD]], {{[a-z0-9]+}}
; OBJ-NEXT: lui [[ADDRESS:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[WORD]], {{(0x)?0}}([[ADDRESS]])
; OBJ: ttsfpstore l0, {{(0x)?4}}, {{(0x)?0}}, {{(0x)?3}}
; OBJ: ret
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %bounded = and i32 %offset, 1023
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 %bounded, i32 3, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

define void @dynamic_full_load_loop(i32 noundef %offset, i32 noundef %count) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_full_load_loop
; ISEL: [[INITIAL:%[0-9]+]]:sfpr = TTSFPLOADFull 0, 0, 3,
; ISEL: :sfpr = PHI [[INITIAL]],
; ISEL: :gpr = PHI
; ISEL: :gpr = PHI
; ISEL: :sfpr, early-clobber {{%[0-9]+}}:gpr, early-clobber {{%[0-9]+}}:gpr = PseudoTTSFPLOADFull {{(killed )?}}{{%[0-9]+}}, 0, 3, implicit
; ISEL: :sfpr = TTSFPADD
; ISEL: BNE
; POST-LABEL: name: dynamic_full_load_loop
; POST: PseudoTTSFPLOADFull
; POST: TTSFPADD
; POST: BNE
; OBJ-LABEL: <dynamic_full_load_loop>:
; OBJ: ttsfpload
; OBJ: beqz
; OBJ: lui [[LOOPWORD:[a-z0-9]+]], 0x70{{[0-7]}}30
; OBJ-NEXT: or [[LOOPWORD]], [[LOOPWORD]], {{[a-z0-9]+}}
; OBJ-NEXT: lui [[LOOPADDRESS:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[LOOPWORD]], {{(0x)?0}}([[LOOPADDRESS]])
; OBJ: ttsfpadd
; OBJ: bnez
; OBJ: ttsfpstore
; OBJ: ret
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 0, i32 0, i32 3)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %sum = phi <32 x i32> [ %initial, %entry ], [ %nextsum, %loop ]
  %coordinate = phi i32 [ %offset, %entry ], [ %next, %loop ]
  %remaining = phi i32 [ %count, %entry ], [ %decremented, %loop ]
  %bounded = and i32 %coordinate, 1023
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 %bounded, i32 0, i32 3)
  %nextsum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %sum, <32 x i32> %sum, <32 x i32> %value, i32 0)
  %next = add i32 %coordinate, 4
  %decremented = add i32 %remaining, -1
  %again = icmp ne i32 %decremented, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %final = phi <32 x i32> [ %initial, %entry ], [ %nextsum, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 8, i32 0, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
