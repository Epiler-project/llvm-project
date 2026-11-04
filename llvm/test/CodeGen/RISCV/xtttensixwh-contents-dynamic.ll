; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.o
; RUN: llvm-objdump -d --mattr=+xtttensixwh %t.o | FileCheck %s --check-prefix=OBJ

; Dynamic issue uses the same Wormhole MC encoding as static issue: AddrMod=3
; occupies bits 15:14, not Blackhole's 15:13. The scalar offset and both
; early-clobber scratch registers stay outside SFPR allocation. The loop is
; retained, including its numerical and scalar PHIs.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dynamic_dst(i32 noundef %offset) "tensix-executor"="trisc1" {
; CHECK-LABEL: dynamic_dst:
; CHECK: lui [[WORD:[a-z0-9]+]], 458812
; CHECK-NEXT: or [[WORD]], [[WORD]], [[OFFSET:[a-z0-9]+]]
; CHECK-NEXT: lui [[ADDRESS:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[WORD]], 0([[ADDRESS]])
; CHECK: lui [[STORE:[a-z0-9]+]], 467004
; CHECK-NEXT: or [[STORE]], [[STORE]], [[OFFSET]]
; CHECK-NEXT: lui [[ADDRESS2:[a-z0-9]+]], 1048128
; CHECK-NEXT: sw [[STORE]], 0([[ADDRESS2]])
; CHECK: ret
; ISEL-LABEL: name: dynamic_dst
; ISEL: [[OFFSET:%[0-9]+]]:gpr = ANDI
; ISEL: [[LOADED:%[0-9]+]]:sfpr, {{.*}}:gpr, {{.*}}:gpr = PseudoTTSFPLOAD {{.*}}, [[OFFSET]], 3, 3
; ISEL: :gpr, {{.*}}:gpr = PseudoTTSFPSTORE {{(killed )?}}[[LOADED]], {{(killed )?}}[[OFFSET]], 3, 3
; OBJ-LABEL: <dynamic_dst>:
; OBJ: lui {{[a-z0-9]+}}, 0x7003c
; OBJ: or
; OBJ: lui {{[a-z0-9]+}}, 0xffe40
; OBJ: sw
; OBJ: lui {{[a-z0-9]+}}, 0x7203c
; OBJ: or
; OBJ: lui {{[a-z0-9]+}}, 0xffe40
; OBJ: sw
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %bounded = and i32 %offset, 1023
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 %bounded, i32 3, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 %bounded, i32 3, i32 3)
  ret void
}

define void @dynamic_dst_loop(i32 noundef %offset, i32 noundef %count) "tensix-executor"="trisc1" {
; CHECK-LABEL: dynamic_dst_loop:
; CHECK: .LBB1_1:
; CHECK: or
; CHECK: sw
; CHECK: or
; CHECK: sw
; CHECK: bne{{z?}} {{.*}}, .LBB1_1
; CHECK: ret
; ISEL-LABEL: name: dynamic_dst_loop
; ISEL: :sfpr = PHI
; ISEL-NEXT: [[INDEX:%[0-9]+]]:gpr = PHI
; ISEL-NEXT: [[REMAINING:%[0-9]+]]:gpr = PHI
; ISEL-NEXT: {{.*}}:gpr = ANDI [[INDEX]], 1023
; ISEL: :sfpr, {{.*}}:gpr, {{.*}}:gpr = PseudoTTSFPLOAD
; ISEL: :gpr, {{.*}}:gpr = PseudoTTSFPSTORE
; ISEL: :gpr = ADDI [[INDEX]], 4
; ISEL-NEXT: [[DECREMENTED:%[0-9]+]]:gpr = ADDI [[REMAINING]], -1
; ISEL-NEXT: BNE [[DECREMENTED]], $x0,
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  br label %loop
loop:
  %value = phi <32 x i32> [ %seed, %0 ], [ %loaded, %loop ]
  %index = phi i32 [ %offset, %0 ], [ %next, %loop ]
  %remaining = phi i32 [ %count, %0 ], [ %decremented, %loop ]
  %bounded = and i32 %index, 1023
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %value, i32 %bounded, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 %bounded, i32 0, i32 3)
  %next = add i32 %index, 4
  %decremented = add i32 %remaining, -1
  %again = icmp ne i32 %decremented, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

; The scalar expansion of a dynamic issue does not replace an SFPU pipeline
; instruction. The tied old destination still reads MAD's result and needs
; the same WH NOP before the dynamic load is pushed to the coprocessor.
define void @dynamic_dst_mad_hazard(i32 noundef %offset) "tensix-executor"="trisc1" {
; OBJ-LABEL: <dynamic_dst_mad_hazard>:
; OBJ: ttsfpmad l0,
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}lui {{[a-z0-9]+}}, 0x70030
; OBJ-NEXT: {{.*}}or
; OBJ-NEXT: {{.*}}lui {{[a-z0-9]+}}, 0xffe40
; OBJ-NEXT: {{.*}}sw
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  %mad = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %seed, <32 x i32> %seed, <32 x i32> %seed, <32 x i32> %seed, i32 0)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %mad, i32 %bounded, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %loaded, i32 %bounded, i32 0, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
