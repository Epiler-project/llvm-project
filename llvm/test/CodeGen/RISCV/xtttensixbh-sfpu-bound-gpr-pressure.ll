; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o %t.isel.mir
; RUN: FileCheck %s --check-prefix=FIXED --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound < %t.isel.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.allocated.mir
; RUN: FileCheck %s --check-prefix=FIXED --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound < %t.allocated.mir
; RUN: FileCheck %s --check-prefix=EFFECTS < %t.allocated.mir
; RUN: FileCheck %s --check-prefix=SPILLS < %t.allocated.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh %t.o | FileCheck %s --check-prefix=NATIVE --implicit-check-not=ttsfpmov --implicit-check-not=ttsfpstore --implicit-check-not=ttsfpload
;
; GPR allocation must preserve the same authored L4/C9 operations under both
; low and overflowing scalar register pressure. All volatile loads precede all
; volatile stores, so the high-pressure body has forty simultaneously live
; scalar values. Reverse stores use every distinct value, preventing DCE or
; incremental reduction from hiding the pressure. Repeated volatile accesses
; to the same pointer are distinct observations, not duplicated loop bodies.
;
; Check real scalar spill/reload instructions, excluding callee-save traffic,
; and retain every volatile effect and the runtime loop. SFPU operands are
; checked both before and after GPR allocation, then in a decoded ELF object.
; A second SFPU allocator, hidden data moves, discarded scalar values, or
; duplicated source iterations must fail independently of the spill policy.
;
; FIXED-LABEL: name: low_gpr_pressure
; FIXED: $tt_l4 = TTSFPMOVAll $tt_c9,
; FIXED: TTSFPSTORE $tt_l4, 0, 0, 4,
; FIXED-LABEL: name: high_gpr_pressure
; FIXED: $tt_l4 = TTSFPMOVAll $tt_c9,
; FIXED: TTSFPSTORE $tt_l4, 0, 0, 4,
;
; EFFECTS-LABEL: name: low_gpr_pressure
; EFFECTS-COUNT-4: volatile load (s32) from %ir.src
; EFFECTS-NOT: volatile load
; EFFECTS-COUNT-4: volatile store (s32) into %ir.dst
; EFFECTS-NOT: volatile store
; EFFECTS-LABEL: name: high_gpr_pressure
; EFFECTS-COUNT-40: volatile load (s32) from %ir.src
; EFFECTS-NOT: volatile load
; EFFECTS-COUNT-40: volatile store (s32) into %ir.dst
; EFFECTS-NOT: volatile store
;
; SPILLS-LABEL: name: low_gpr_pressure
; SPILLS-NOT: type: spill-slot
; SPILLS: body:
; SPILLS-NOT: %stack.
; SPILLS-LABEL: name: high_gpr_pressure
; SPILLS: type: spill-slot
; SPILLS: body:
; SPILLS: {{^ *}}SW {{.*}}%stack.
; SPILLS: {{^ *}}renamable ${{x[0-9]+}} = LW {{.*}} :: (load (s32) from %stack.
;
; NATIVE-LABEL: <low_gpr_pressure>:
; NATIVE: [[LOW_HEAD:[0-9a-f]+]]: {{[0-9a-f]+}} lw
; NATIVE: ttsfpmov.all l4, c9
; NATIVE: ttsfpstore l4, 0, 0, 4
; NATIVE: {{bne|bnez|bltu|blt}} {{.*}}0x[[LOW_HEAD]] <low_gpr_pressure+0x{{[0-9a-f]+}}>
; NATIVE: ret
; NATIVE-LABEL: <high_gpr_pressure>:
; NATIVE: [[HIGH_HEAD:[0-9a-f]+]]: {{[0-9a-f]+}} lw
; NATIVE: ttsfpmov.all l4, c9
; NATIVE: ttsfpstore l4, 0, 0, 4
; NATIVE: {{bne|bnez|bltu|blt}} {{.*}}0x[[HIGH_HEAD]] <high_gpr_pressure+0x{{[0-9a-f]+}}>
; NATIVE: ret

declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @low_gpr_pressure(ptr %src, ptr %dst, i32 %count) "tensix-executor"="trisc1" {
entry:
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  %v0 = load volatile i32, ptr %src, align 4
  %v1 = load volatile i32, ptr %src, align 4
  %v2 = load volatile i32, ptr %src, align 4
  %v3 = load volatile i32, ptr %src, align 4
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 0, i32 0, i32 4)
  store volatile i32 %v3, ptr %dst, align 4
  store volatile i32 %v2, ptr %dst, align 4
  store volatile i32 %v1, ptr %dst, align 4
  store volatile i32 %v0, ptr %dst, align 4
  %next = add i32 %remaining, -1
  %more = icmp ne i32 %next, 0
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

define void @high_gpr_pressure(ptr %src, ptr %dst, i32 %count) "tensix-executor"="trisc1" {
entry:
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  %v0 = load volatile i32, ptr %src, align 4
  %v1 = load volatile i32, ptr %src, align 4
  %v2 = load volatile i32, ptr %src, align 4
  %v3 = load volatile i32, ptr %src, align 4
  %v4 = load volatile i32, ptr %src, align 4
  %v5 = load volatile i32, ptr %src, align 4
  %v6 = load volatile i32, ptr %src, align 4
  %v7 = load volatile i32, ptr %src, align 4
  %v8 = load volatile i32, ptr %src, align 4
  %v9 = load volatile i32, ptr %src, align 4
  %v10 = load volatile i32, ptr %src, align 4
  %v11 = load volatile i32, ptr %src, align 4
  %v12 = load volatile i32, ptr %src, align 4
  %v13 = load volatile i32, ptr %src, align 4
  %v14 = load volatile i32, ptr %src, align 4
  %v15 = load volatile i32, ptr %src, align 4
  %v16 = load volatile i32, ptr %src, align 4
  %v17 = load volatile i32, ptr %src, align 4
  %v18 = load volatile i32, ptr %src, align 4
  %v19 = load volatile i32, ptr %src, align 4
  %v20 = load volatile i32, ptr %src, align 4
  %v21 = load volatile i32, ptr %src, align 4
  %v22 = load volatile i32, ptr %src, align 4
  %v23 = load volatile i32, ptr %src, align 4
  %v24 = load volatile i32, ptr %src, align 4
  %v25 = load volatile i32, ptr %src, align 4
  %v26 = load volatile i32, ptr %src, align 4
  %v27 = load volatile i32, ptr %src, align 4
  %v28 = load volatile i32, ptr %src, align 4
  %v29 = load volatile i32, ptr %src, align 4
  %v30 = load volatile i32, ptr %src, align 4
  %v31 = load volatile i32, ptr %src, align 4
  %v32 = load volatile i32, ptr %src, align 4
  %v33 = load volatile i32, ptr %src, align 4
  %v34 = load volatile i32, ptr %src, align 4
  %v35 = load volatile i32, ptr %src, align 4
  %v36 = load volatile i32, ptr %src, align 4
  %v37 = load volatile i32, ptr %src, align 4
  %v38 = load volatile i32, ptr %src, align 4
  %v39 = load volatile i32, ptr %src, align 4
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 0, i32 0, i32 4)
  store volatile i32 %v39, ptr %dst, align 4
  store volatile i32 %v38, ptr %dst, align 4
  store volatile i32 %v37, ptr %dst, align 4
  store volatile i32 %v36, ptr %dst, align 4
  store volatile i32 %v35, ptr %dst, align 4
  store volatile i32 %v34, ptr %dst, align 4
  store volatile i32 %v33, ptr %dst, align 4
  store volatile i32 %v32, ptr %dst, align 4
  store volatile i32 %v31, ptr %dst, align 4
  store volatile i32 %v30, ptr %dst, align 4
  store volatile i32 %v29, ptr %dst, align 4
  store volatile i32 %v28, ptr %dst, align 4
  store volatile i32 %v27, ptr %dst, align 4
  store volatile i32 %v26, ptr %dst, align 4
  store volatile i32 %v25, ptr %dst, align 4
  store volatile i32 %v24, ptr %dst, align 4
  store volatile i32 %v23, ptr %dst, align 4
  store volatile i32 %v22, ptr %dst, align 4
  store volatile i32 %v21, ptr %dst, align 4
  store volatile i32 %v20, ptr %dst, align 4
  store volatile i32 %v19, ptr %dst, align 4
  store volatile i32 %v18, ptr %dst, align 4
  store volatile i32 %v17, ptr %dst, align 4
  store volatile i32 %v16, ptr %dst, align 4
  store volatile i32 %v15, ptr %dst, align 4
  store volatile i32 %v14, ptr %dst, align 4
  store volatile i32 %v13, ptr %dst, align 4
  store volatile i32 %v12, ptr %dst, align 4
  store volatile i32 %v11, ptr %dst, align 4
  store volatile i32 %v10, ptr %dst, align 4
  store volatile i32 %v9, ptr %dst, align 4
  store volatile i32 %v8, ptr %dst, align 4
  store volatile i32 %v7, ptr %dst, align 4
  store volatile i32 %v6, ptr %dst, align 4
  store volatile i32 %v5, ptr %dst, align 4
  store volatile i32 %v4, ptr %dst, align 4
  store volatile i32 %v3, ptr %dst, align 4
  store volatile i32 %v2, ptr %dst, align 4
  store volatile i32 %v1, ptr %dst, align 4
  store volatile i32 %v0, ptr %dst, align 4
  %next = add i32 %remaining, -1
  %more = icmp ne i32 %next, 0
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
