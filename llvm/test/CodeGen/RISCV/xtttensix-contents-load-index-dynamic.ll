; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "

; A dynamic coordinate uses exactly the ordinary Dst instruction-port word.
; Both actual numerical destinations and their tied olds remain visible until
; allocation, together with independent early-clobber word/address scratch.
; The loop retains both numerical PHIs, its scalar coordinate, and zero trip.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dynamic_pair(i32 noundef %offset) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_pair
; ISEL: :sfprindexpair, {{.*}}:gpr, {{.*}}:gpr = PseudoTTSFPContentsLoadIndexDynamic {{.*}}, {{.*}}, 0, 3,
; POST-LABEL: name: dynamic_pair
; POST: {{(renamable )?}}$tt_l0, {{(renamable )?}}$tt_l4, {{(dead )?}}early-clobber {{(renamable )?}}${{[a-z0-9]+}}, {{(dead )?}}early-clobber {{(renamable )?}}${{[a-z0-9]+}} = PseudoTTSFPLOADIndex {{(killed )?(renamable )?}}$tt_l0, {{(killed )?(renamable )?}}$tt_l4, {{(killed )?(renamable )?}}${{[a-z0-9]+}}, 0, 3,
; POST-SAME: :: (volatile load store (s32))
; OBJ-LABEL: <dynamic_pair>:
; OBJ: lui [[WORD:[a-z0-9]+]], 0x70030
; OBJ-NEXT: or [[WORD]], [[WORD]], [[OFFSET:[a-z0-9]+]]
; OBJ-NEXT: lui [[ADDRESS:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[WORD]], {{(0x)?0}}([[ADDRESS]])
; OBJ: ttsfpstore l0,
; OBJ: ttsfpstore l4,
; OBJ: ret
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  %bounded = and i32 %offset, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 %bounded, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

define void @dynamic_pair_loop(i32 noundef %offset, i32 noundef %count) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: dynamic_pair_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = PHI
; ISEL: :gpr = PHI
; ISEL: :sfprindexpair, {{.*}}:gpr, {{.*}}:gpr = PseudoTTSFPContentsLoadIndexDynamic
; ISEL: BNE
; POST-LABEL: name: dynamic_pair_loop
; POST: PseudoTTSFPLOADIndex
; POST: BNE
; OBJ-LABEL: <dynamic_pair_loop>:
; OBJ: beqz
; OBJ: lui [[LOOPWORD:[a-z0-9]+]], 0x70030
; OBJ-NEXT: or [[LOOPWORD]], [[LOOPWORD]], {{[a-z0-9]+}}
; OBJ-NEXT: lui [[LOOPADDRESS:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[LOOPWORD]], {{(0x)?0}}([[LOOPADDRESS]])
; OBJ: bnez
; OBJ: ttsfpstore l0,
; OBJ: ttsfpstore l4,
; OBJ: ret
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %initial, i32 21845, i32 8)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %oldvalue = phi <32 x i32> [ %initial, %entry ], [ %value, %loop ]
  %oldindex = phi <32 x i32> [ %index, %entry ], [ %captured, %loop ]
  %coordinate = phi i32 [ %offset, %entry ], [ %next, %loop ]
  %remaining = phi i32 [ %count, %entry ], [ %decremented, %loop ]
  %bounded = and i32 %coordinate, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %oldvalue, <32 x i32> %oldindex, i32 %bounded, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  %next = add i32 %coordinate, 4
  %decremented = add i32 %remaining, -1
  %again = icmp ne i32 %decremented, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %finalvalue = phi <32 x i32> [ %initial, %entry ], [ %value, %loop ]
  %finalindex = phi <32 x i32> [ %index, %entry ], [ %captured, %loop ]
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %finalvalue, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %finalindex, i32 2, i32 0, i32 4)
  ret void
}

define void @dynamic_pair_addr3(i32 noundef %offset) "tensix-executor"="trisc1" {
; OBJ-LABEL: <dynamic_pair_addr3>:
; BH: lui [[ADDR3WORD:[a-z0-9]+]], 0x70036
; WH: lui [[ADDR3WORD:[a-z0-9]+]], 0x7003c
; OBJ-NEXT: or [[ADDR3WORD]], [[ADDR3WORD]], {{[a-z0-9]+}}
; OBJ-NEXT: lui [[ADDR3PORT:[a-z0-9]+]], 0xffe40
; OBJ-NEXT: sw [[ADDR3WORD]], {{(0x)?0}}([[ADDR3PORT]])
; OBJ: ttsfpstore l0,
; OBJ: ttsfpstore l4,
; OBJ: ret
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %iseed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %iseed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  %bounded = and i32 %offset, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 %bounded, i32 3, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
