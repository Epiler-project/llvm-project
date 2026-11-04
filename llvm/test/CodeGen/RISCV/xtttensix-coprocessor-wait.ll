; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=MI
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=MI
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=MI
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/valid.ll -o - | FileCheck %s --check-prefix=MI
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/canonical-declaration.ll -o - | FileCheck %s --check-prefix=CANONICAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %t/canonical-declaration.ll -o - | FileCheck %s --check-prefix=CANONICAL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/ncrisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/ncrisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/brisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/brisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/missing-executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/missing-executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/memory.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/memory.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MEMORY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/nosync.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SYNC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/nosync.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SYNC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/bundle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/bundle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/callingconv.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/callingconv.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -O2 %t/valid.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv64 -mattr=+xtttensixbh -O2 %t/valid.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV32
; RUN: not llc -mtriple=riscv64 -mattr=+xtttensixwh -O2 %t/valid.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV32

; The wait is a single ordered, nonduplicable machine owner until emission.
; Both scratch registers belong to the ordinary GPR allocator. The four fixed
; instructions enforce store ordering, observe coprocessor execution (not only
; MOP expansion), and immediately consume the returned GPR before another load.
; This hardware primitive does not disable PCBuf overrides owned by the caller.
; ASM-LABEL: wait_trisc0:
; ASM: lui [[ADDRESS:[a-z][a-z0-9]*]], 1048192
; ASM-NEXT: sw zero, 4([[ADDRESS]])
; ASM-NEXT: lw [[RESPONSE:[a-z][a-z0-9]*]], 4([[ADDRESS]])
; ASM-NEXT: mv zero, [[RESPONSE]]
; ASM-LABEL: wait_trisc1:
; ASM: lui [[ADDRESS:[a-z][a-z0-9]*]], 1048192
; ASM-NEXT: sw zero, 4([[ADDRESS]])
; ASM-NEXT: lw [[RESPONSE:[a-z][a-z0-9]*]], 4([[ADDRESS]])
; ASM-NEXT: mv zero, [[RESPONSE]]
; ASM-LABEL: wait_trisc2:
; ASM: lui [[ADDRESS:[a-z][a-z0-9]*]], 1048192
; ASM-NEXT: sw zero, 4([[ADDRESS]])
; ASM-NEXT: lw [[RESPONSE:[a-z][a-z0-9]*]], 4([[ADDRESS]])
; ASM-NEXT: mv zero, [[RESPONSE]]
; MI-COUNT-3: early-clobber %{{[0-9]+}}:gpr, {{(dead )?}}early-clobber %{{[0-9]+}}:gpr = PseudoTTCoprocessorWait implicit-def{{( dead)?}} $tt_issue, implicit $tt_issue
; CANONICAL-LABEL: canonical_declaration:
; CANONICAL: lui [[ADDRESS:[a-z][a-z0-9]*]], 1048192
; CANONICAL-NEXT: sw zero, 4([[ADDRESS]])
; CANONICAL-NEXT: lw [[RESPONSE:[a-z][a-z0-9]*]], 4([[ADDRESS]])
; CANONICAL-NEXT: mv zero, [[RESPONSE]]
; EXECUTOR: coprocessor wait requires a TRISC executor
; MISSING: Tensix intrinsic requires a tensix-executor function attribute
; MEMORY: coprocessor wait requires full memory effects
; SYNC: coprocessor wait requires convergent nonduplicable synchronization
; CALL: {{(Tensix intrinsic|coprocessor wait)}} requires a direct C call without operand bundles
; FEATURE: coprocessor wait requires +xtttensixbh or +xtttensixwh
; RV32: XTTTensix{{BH|WH}} requires RV32 little-endian

;--- valid.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @wait_trisc0() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}
define void @wait_trisc1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}
define void @wait_trisc2() "tensix-executor"="trisc2" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}

;--- ncrisc.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}

;--- brisc.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}

;--- missing-executor.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe()  {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}

; LLVM restores intrinsic declaration attributes from TableGen. This cannot
; weaken the synchronization owner; the complete wait must still be emitted.
;--- canonical-declaration.ll
declare void @llvm.riscv.tt.coprocessor.wait() memory(none) nosync
define void @canonical_declaration() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}

;--- memory.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.coprocessor.wait() memory(none)
  ret void
}

;--- nosync.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.coprocessor.wait() nosync
  ret void
}

;--- bundle.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.coprocessor.wait() [ "deopt"(i32 0) ]
  ret void
}

;--- callingconv.ll
declare void @llvm.riscv.tt.coprocessor.wait()
define void @probe() "tensix-executor"="trisc1" {
  call fastcc void @llvm.riscv.tt.coprocessor.wait()
  ret void
}
