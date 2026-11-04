; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=call --implicit-check-not=tail --implicit-check-not=fence --implicit-check-not=csrr
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=call --implicit-check-not=tail --implicit-check-not=fence --implicit-check-not=csrr
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=call --implicit-check-not=tail --implicit-check-not=fence --implicit-check-not=csrr
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stop-after=machine-scheduler -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR
;
; A scoped exact-width read clobbers only its MLIR-bound scratch group and preserves
; an active setvl result. With no source configuration it establishes e32/VL0.
declare i64 @llvm.riscv.bosc.read.bits(i32 immarg, ptr, i64, i32 immarg)
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; ASM-LABEL: read_byte:
; ASM: li [[VL:[a-z][0-9]+]], 0
; ASM: vsetivli zero, 1, e8, m1, ta, ma
; ASM-NEXT: vle8.v v31, ({{[a-z][0-9]+}})
; ASM-NEXT: vmv.x.s [[BYTE:[a-z][0-9]+]], v31
; ASM-NEXT: zext.b [[BYTE]], [[BYTE]]
; ASM-NEXT: vsetvli zero, [[VL]], e32, m1, ta, ma
; ASM: ret
; MIR-LABEL: name: read_byte
; MIR: PseudoBOSCReadBitsBound
; MIR-SAME: implicit $bosc_issue, implicit-def $bosc_issue
; MIR-SAME: :: (volatile load (s8)
define i64 @read_byte(ptr %source) "trex-bosc-bound-rvv"="1" {
  %byte = call i64 @llvm.riscv.bosc.read.bits(i32 31, ptr %source, i64 0, i32 8)
  ret i64 %byte
}

; A dynamic configuration's initialized vector contents remain readable after
; the temporary observation configuration, including at CFG joins and loop backedges.
; ASM-LABEL: live_dynamic_vector:
; ASM: vsetvli [[VL2:[a-z][0-9]+]], {{[a-z][0-9]+}}, e32, m1, ta, ma
; ASM: vle32.v v4,
; ASM: vsetivli zero, 1, e8, m1, ta, ma
; ASM-NEXT: vle8.v v7,
; ASM-NEXT: vmv.x.s [[BYTE2:[a-z][0-9]+]], v7
; ASM-NEXT: zext.b [[BYTE2]], [[BYTE2]]
; ASM-NEXT: vsetvli zero, [[VL2]], e32, m1, ta, ma
; ASM: vse32.v v4,
; ASM: ret
define i64 @live_dynamic_vector(ptr %input, ptr %output, i64 %avl)
    "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  call void @llvm.riscv.bosc.load(i32 4, ptr %input)
  %byte = call i64 @llvm.riscv.bosc.read.bits(i32 7, ptr %input, i64 %vl, i32 8)
  call void @llvm.riscv.bosc.store(i32 4, ptr %output)
  ret i64 %byte
}

; ASM-LABEL: word_loop:
; ASM: [[LOOP:.LBB[0-9]+_[0-9]+]]:
; ASM: vle32.v v9,
; ASM: vsetvli zero, {{[a-z][0-9]+}}, e32, m1, ta, ma
; ASM: bne{{z?}} {{.*}}[[LOOP]]
; ASM: ret
define i64 @word_loop(ptr %input, i64 %count) "trex-bosc-bound-rvv"="1" {
entry:
  %empty = icmp eq i64 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %sum = phi i64 [ 0, %entry ], [ %added, %loop ]
  %at = getelementptr i32, ptr %input, i64 %i
  %byte = call i64 @llvm.riscv.bosc.read.bits(i32 9, ptr %at, i64 0, i32 32)
  %added = add i64 %sum, %byte
  %next = add i64 %i, 1
  %more = icmp ne i64 %next, %count
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  %result = phi i64 [ 0, %entry ], [ %added, %loop ]
  ret i64 %result
}

; Discarding the result cannot erase the observation or its restoration.
; ASM-LABEL: unused_byte:
; ASM: li [[VL3:[a-z][0-9]+]], 0
; ASM: vsetivli zero, 1, e8, m1, ta, ma
; ASM-NEXT: vle8.v v2,
; ASM-NEXT: vmv.x.s [[BYTE3:[a-z][0-9]+]], v2
; ASM-NEXT: zext.b [[BYTE3]], [[BYTE3]]
; ASM-NEXT: vsetvli zero, [[VL3]], e32, m1, ta, ma
; ASM: ret
define void @unused_byte(ptr %input) "trex-bosc-bound-rvv"="1" {
  %unused = call i64 @llvm.riscv.bosc.read.bits(i32 2, ptr %input, i64 0, i32 8)
  ret void
}

; Zero is restored from the actual setvl result in a nonzero GPR, not by using
; the special x0 AVL encoding or substituting the original request.
; ASM-LABEL: active_zero:
; ASM: vsetivli [[ZERO:[a-z][0-9]+]], 0, e32, m1, ta, ma
; ASM: vsetivli zero, 1, e8, m1, ta, ma
; ASM: vle8.v v5,
; ASM: vsetvli zero, [[ZERO]], e32, m1, ta, ma
; ASM: ret
define i64 @active_zero(ptr %input) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 0)
  %byte = call i64 @llvm.riscv.bosc.read.bits(i32 5, ptr %input, i64 %vl, i32 8)
  ret i64 %byte
}


; A signed-looking f32/i32 bit pattern is returned zero extended, not as an
; XLEN-sign-extended value. The access is one aligned word, never four bytes.
; ASM-LABEL: read_word:
; ASM: li [[WORDVL:[a-z][0-9]+]], 0
; ASM: vsetivli zero, 1, e32, m1, ta, ma
; ASM-NEXT: vle32.v v6, ({{[a-z][0-9]+}})
; ASM-NEXT: vmv.x.s [[WORD:[a-z][0-9]+]], v6
; ASM-NEXT: slli [[WORD]], [[WORD]], 32
; ASM-NEXT: srli [[WORD]], [[WORD]], 32
; ASM-NEXT: vsetvli zero, [[WORDVL]], e32, m1, ta, ma
; ASM: ret
; MIR-LABEL: name: read_word
; MIR: PseudoBOSCReadBitsBound {{.*}}, 32,
; MIR-SAME: :: (volatile load (s32)
define i64 @read_word(ptr %source) "trex-bosc-bound-rvv"="1" {
  %bits = call i64 @llvm.riscv.bosc.read.bits(i32 6, ptr %source, i64 0, i32 32)
  ret i64 %bits
}

; An i64/f64 payload uses one e64 lane and keeps all 64 source bits. The
; temporary e64 state must restore the actual active e32 VL, not the AVL.
; ASM-LABEL: read_doubleword:
; ASM: vsetvli [[DWORDVL:[a-z][0-9]+]], {{[a-z][0-9]+}}, e32, m1, ta, ma
; ASM: vle32.v v4,
; ASM: vsetivli zero, 1, e64, m1, ta, ma
; ASM-NEXT: vle64.v v8, ({{[a-z][0-9]+}})
; ASM-NEXT: vmv.x.s [[DWORD:[a-z][0-9]+]], v8
; ASM-NEXT: vsetvli zero, [[DWORDVL]], e32, m1, ta, ma
; ASM: vse32.v v4,
; ASM: ret
; MIR-LABEL: name: read_doubleword
; MIR: PseudoBOSCReadBitsBound {{.*}}, 64,
; MIR-SAME: :: (volatile load (s64)
define i64 @read_doubleword(ptr %source, ptr %out, i64 %avl)
    "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  call void @llvm.riscv.bosc.load(i32 4, ptr %source)
  %bits = call i64 @llvm.riscv.bosc.read.bits(i32 8, ptr %source, i64 %vl, i32 64)
  call void @llvm.riscv.bosc.store(i32 4, ptr %out)
  ret i64 %bits
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
