; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=IMPLICIT_DEF
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY " --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPENTRY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY " --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPENTRY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY " --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPENTRY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY " --implicit-check-not=IMPLICIT_DEF --implicit-check-not=TTSFPENTRY
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,OBJ0 --implicit-check-not="ttsfp{{load|mov}}"
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefixes=OBJ,OBJ2 --implicit-check-not="ttsfp{{load|mov}}"
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,OBJ0 --implicit-check-not="ttsfp{{load|mov}}"
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefixes=OBJ,OBJ2 --implicit-check-not="ttsfp{{load|mov}}"

; An entry is one stable observation of arbitrary hardware bits in an ordinary
; allocated LReg. It has no selector and does not initialize any payload bits.
; The mandatory receiver must retain its actual identity without requiring the
; optional initialized-payload proof. Native setup below is authored separately.
; In particular, no constant snapshot, load, move or generic implicit definition
; may substitute for this observation.
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

; Both unknown entries are real inputs, including the tied old destination.
; Their zero-size definitions must survive ISel and RA; after RA the selected
; hardware values must be real block live-ins rather than undefined uses.
define void @entry_unknown_add_mul() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: entry_unknown_add_mul
; ISEL: [[A:%[0-9]+]]:sfpr = TTSFPENTRY
; ISEL: [[B:%[0-9]+]]:sfpr = TTSFPENTRY
; ISEL: [[SUM:%[0-9]+]]:sfpr = TTSFPADD [[A]], [[A]], [[B]], 0
; ISEL: :sfpr = TTSFPMUL [[SUM]], [[SUM]], [[B]], 0
; POST-LABEL: name: entry_unknown_add_mul
; POST: stackSize: 0
; POST: liveins: {{.*}}$tt_l{{[0-7]}}
; POST: TTSFPADD
; POST: TTSFPMUL
; POST: TTSFPSTORE
; OBJ-LABEL: <entry_unknown_add_mul>:
; OBJ: ttsfpencc 3, 10
; OBJ-NEXT: {{.*}}ttsfpconfig.reset
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpadd [[A:l[0-7]]], [[A]], [[B:l[0-7]]], 0
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpmul [[A]], [[A]], [[B]], 0
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfpstore [[A]], 0, 0, 3
; OBJ-NOT: ttsfp
; OBJ: ret
entry:
  %first = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %second = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %first, <32 x i32> %first, <32 x i32> %second, i32 0)
  call void @llvm.riscv.tt.contents.nop()
  %product = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %sum, <32 x i32> %sum, <32 x i32> %second, i32 0)
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %product, i32 0, i32 0, i32 3)
  ret void
}

; A partial first write preserves the observed high bits and inactive lanes.
; State-only setup does not end the entry prefix or supply numerical contents.
; Replacing the old value with an initialized seed would change this program.
define void @entry_partial_preserves_observed_old() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: entry_partial_preserves_observed_old
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPENTRY
; ISEL: :sfpr = TTSFPLOADI [[OLD]], 4660, 10
; POST-LABEL: name: entry_partial_preserves_observed_old
; POST: stackSize: 0
; POST: liveins: {{.*}}$tt_l{{[0-7]}}
; POST: TTSFPLOADI
; POST: TTSFPSTORE
; OBJ-LABEL: <entry_partial_preserves_observed_old>:
; OBJ: ttsfpencc 3, 10
; OBJ-NEXT: {{.*}}ttsfpconfig.reset
; OBJ-NEXT: {{.*}}ttsfpnop
; OBJ-NEXT: {{.*}}ttsfploadi [[PARTIAL:l[0-7]]], 4660, 10
; OBJ-NEXT: {{.*}}ttsfpstore [[PARTIAL]], 0, 0, 3
; OBJ-NOT: ttsfp
; OBJ: ret
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %observed = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %observed, i32 4660, i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %partial, i32 0, i32 0, i32 3)
  ret void
}

; At O0 ordinary GPR counters may spill. The numerical SFPRs must remain
; tied physical live-ins with no register copies or numerical spill/load.
; O2 may rotate the authored tail NOP to the header; its backedge must reach
; that NOP before the next branch, including the final exit test.
; The entry definition precedes the loop. The actual PHI transports the latest
; version on its backedge and retains the observed value on the zero-trip edge.
; A seed in the body, re-created iterations or a repair copy cannot implement it.
define void @entry_zero_trip_loop(i32 %limit) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: entry_zero_trip_loop
; ISEL: [[INITIAL:%[0-9]+]]:sfpr = TTSFPENTRY
; ISEL: [[STEP:%[0-9]+]]:sfpr = TTSFPENTRY
; ISEL: bb.[[LOOP:[0-9]+]].loop:
; ISEL: [[CURRENT:%[0-9]+]]:sfpr = PHI [[INITIAL]], %bb.{{[0-9]+}}, [[NEXT:%[0-9]+]], %bb.[[LOOP]]
; ISEL: [[NEXT]]:sfpr = TTSFPADD [[CURRENT]], [[CURRENT]], [[STEP]], 0
; ISEL: [[EXIT:%[0-9]+]]:sfpr = PHI [[INITIAL]], %bb.0, [[NEXT]], %bb.[[LOOP]]
; ISEL: TTSFPSTORE [[EXIT]], 0, 0, 3
; POST-LABEL: name: entry_zero_trip_loop
; POST: bb.0.entry:
; POST: liveins: $tt_l[[ACC:[0-7]]], $tt_l[[STEP:[0-7]]]
; POST: bb.{{[0-9]+}}.loop:
; POST: liveins: $tt_l[[ACC]], $tt_l[[STEP]]
; POST: $tt_l[[ACC]] = TTSFPADD {{(killed )?(renamable )?}}$tt_l[[ACC]], {{(renamable )?}}$tt_l[[ACC]], {{(renamable )?}}$tt_l[[STEP]], 0
; POST-NOT: TTSFPADD
; POST: bb.{{[0-9]+}}.exit:
; POST: liveins: $tt_l[[ACC]]
; POST: TTSFPSTORE {{(killed )?(renamable )?}}$tt_l[[ACC]], 0, 0, 3
; OBJ-LABEL: <entry_zero_trip_loop>:
; OBJ: ttsfpencc 3, 10
; OBJ-NEXT: {{.*}}ttsfpconfig.reset
; OBJ-NEXT: [[HEAD:[0-9a-f]+]]: {{[0-9a-f]+}} ttsfpnop
; OBJ: beqz {{[a-z0-9]+}}, 0x[[EXIT:[0-9a-f]+]]
; OBJ0: ttsfpadd [[ACC:l[0-7]]], [[ACC]], [[STEP:l[0-7]]], 0
; OBJ0-NEXT: {{.*}}ttsfpnop
; OBJ0-NOT: ttsfpadd
; OBJ0: bne {{.*}},
; OBJ0-NEXT: {{.*}}j 0x[[EXIT]]
; OBJ2-NEXT: {{.*}}ttsfpadd [[ACC:l[0-7]]], [[ACC]], [[STEP:l[0-7]]], 0
; OBJ2-NEXT: {{.*}}addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, -1
; OBJ2-NEXT: {{.*}}j 0x[[HEAD]]
; OBJ-NOT: ttsfpadd
; OBJ: [[EXIT]]: {{[0-9a-f]+}} ttsfpstore [[ACC]], 0, 0, 3
; OBJ-NOT: ttsfp
; OBJ: ret
entry:
  %initial = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %step = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %empty = icmp eq i32 %limit, 0
  br i1 %empty, label %exit, label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %current = phi <32 x i32> [ %initial, %entry ], [ %sum, %loop ]
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %current, <32 x i32> %current, <32 x i32> %step, i32 0)
  call void @llvm.riscv.tt.contents.nop()
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %result = phi <32 x i32> [ %initial, %entry ], [ %sum, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 3)
  ret void
}

; Observing unused contents neither issues numerical work nor writes storage.
define void @entry_observation_only() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: entry_observation_only
; ISEL-NOT: TTSFP{{ADD|MUL|MOV|LOAD|STORE}}
; POST-LABEL: name: entry_observation_only
; POST: stackSize: 0
; POST-NOT: TTSFP
; OBJ-LABEL: <entry_observation_only>:
; OBJ-NOT: ttsfp
; OBJ: ret
entry:
  %observed = call <32 x i32> @llvm.riscv.tt.contents.entry()
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
