; Zaamo-only support must not make unsupported atomics reach instruction
; selection or create malformed masked LR/SC intrinsic calls. Native load/store
; and AMO operations retain their ordering; unsupported RMW/CAS use the generic
; LLVM atomic library ABI. This does not add that ABI to a device firmware.
;
; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+zaamo,-zalrsc -verify-machineinstrs %t/load-store.ll -o - | FileCheck %s --check-prefix=MEM
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc -verify-machineinstrs %t/load-store.ll -o - | FileCheck %s --check-prefix=MEM
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc -verify-machineinstrs %t/double.ll -o - | FileCheck %s --check-prefix=DOUBLE
; RUN: llc -mtriple=riscv32 -mattr=+zaamo,-zalrsc,-zacas -verify-machineinstrs %t/rmw-cas.ll -o - | FileCheck %s --check-prefixes=COMMON,LIB --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc,-zacas -verify-machineinstrs %t/rmw-cas.ll -o - | FileCheck %s --check-prefixes=COMMON,LIB --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv32 -mattr=+a -verify-machineinstrs %t/rmw-cas.ll -o - | FileCheck %s --check-prefixes=COMMON,LRSC --implicit-check-not=__sync --implicit-check-not=__atomic
; RUN: llc -mtriple=riscv64 -mattr=+a -verify-machineinstrs %t/rmw-cas.ll -o - | FileCheck %s --check-prefixes=COMMON,LRSC --implicit-check-not=__sync --implicit-check-not=__atomic
; RUN: llc -mtriple=riscv32 -mattr=+zalrsc,-zaamo -verify-machineinstrs %t/rmw-cas.ll -o - | FileCheck %s --check-prefixes=COMMON,LRSC --implicit-check-not=__sync --implicit-check-not=__atomic
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,+zacas,-zalrsc -verify-machineinstrs %t/zacas.ll -o - | FileCheck %s --check-prefix=ZACAS --implicit-check-not=__sync --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.

;--- load-store.ll
; MEM-LABEL: load8:
; MEM: fence rw, rw
; MEM: lb a0, 0(a0)
; MEM: fence r, rw
; MEM: ret
define i8 @load8(ptr %p) {
  %v = load atomic i8, ptr %p seq_cst, align 1
  ret i8 %v
}

; MEM-LABEL: load16:
; MEM: lh a0, 0(a0)
; MEM: fence r, rw
; MEM: ret
define i16 @load16(ptr %p) {
  %v = load atomic i16, ptr %p acquire, align 2
  ret i16 %v
}

; MEM-LABEL: load32:
; MEM: lw a0, 0(a0)
; MEM: ret
define i32 @load32(ptr %p) {
  %v = load atomic i32, ptr %p monotonic, align 4
  ret i32 %v
}

; MEM-LABEL: store8:
; MEM: fence rw, w
; MEM: sb a1, 0(a0)
; MEM: ret
define void @store8(ptr %p, i8 %v) {
  store atomic i8 %v, ptr %p release, align 1
  ret void
}

; MEM-LABEL: store16:
; MEM: fence rw, w
; MEM: sh a1, 0(a0)
; MEM: ret
define void @store16(ptr %p, i16 %v) {
  store atomic i16 %v, ptr %p seq_cst, align 2
  ret void
}

; MEM-LABEL: store32:
; MEM: sw a1, 0(a0)
; MEM: ret
define void @store32(ptr %p, i32 %v) {
  store atomic i32 %v, ptr %p monotonic, align 4
  ret void
}

;--- double.ll
; DOUBLE-LABEL: load64:
; DOUBLE: fence rw, rw
; DOUBLE: ld a0, 0(a0)
; DOUBLE: fence r, rw
; DOUBLE: ret
define i64 @load64(ptr %p) {
  %v = load atomic i64, ptr %p seq_cst, align 8
  ret i64 %v
}

; DOUBLE-LABEL: store64:
; DOUBLE: fence rw, w
; DOUBLE: sd a1, 0(a0)
; DOUBLE: ret
define void @store64(ptr %p, i64 %v) {
  store atomic i64 %v, ptr %p release, align 8
  ret void
}

;--- rmw-cas.ll
; COMMON-LABEL: add8:
; LIB: call __sync_fetch_and_add_1
; LRSC: lr.w
; LRSC: sc.w
; COMMON: ret
define i8 @add8(ptr %p, i8 %v) {
  %old = atomicrmw add ptr %p, i8 %v seq_cst, align 1
  ret i8 %old
}

; COMMON-LABEL: xchg16:
; LIB: call __sync_lock_test_and_set_2
; LRSC: lr.w
; LRSC: sc.w
; COMMON: ret
define i16 @xchg16(ptr %p, i16 %v) {
  %old = atomicrmw xchg ptr %p, i16 %v seq_cst, align 2
  ret i16 %old
}

; COMMON-LABEL: nand32:
; LIB: call __sync_fetch_and_nand_4
; LRSC: lr.w
; LRSC: sc.w
; COMMON: ret
define i32 @nand32(ptr %p, i32 %v) {
  %old = atomicrmw nand ptr %p, i32 %v seq_cst, align 4
  ret i32 %old
}

; COMMON-LABEL: cas8:
; LIB: call __sync_val_compare_and_swap_1
; LRSC: lr.w
; LRSC: sc.w
; COMMON: ret
define {i8, i1} @cas8(ptr %p, i8 %e, i8 %v) {
  %old = cmpxchg ptr %p, i8 %e, i8 %v seq_cst seq_cst, align 1
  ret {i8, i1} %old
}

; COMMON-LABEL: cas32:
; LIB: call __sync_val_compare_and_swap_4
; LRSC: lr.w
; LRSC: sc.w
; COMMON: ret
define {i32, i1} @cas32(ptr %p, i32 %e, i32 %v) {
  %old = cmpxchg ptr %p, i32 %e, i32 %v seq_cst seq_cst, align 4
  ret {i32, i1} %old
}

;--- zacas.ll
; ZACAS-LABEL: cas32:
; ZACAS: amocas.w.aqrl
; ZACAS: ret
define {i32, i1} @cas32(ptr %p, i32 %e, i32 %v) {
  %old = cmpxchg ptr %p, i32 %e, i32 %v seq_cst seq_cst, align 4
  ret {i32, i1} %old
}

; ZACAS-LABEL: nand32:
; ZACAS: amocas.w.aqrl
; ZACAS: ret
define i32 @nand32(ptr %p, i32 %v) {
  %old = atomicrmw nand ptr %p, i32 %v seq_cst, align 4
  ret i32 %old
}
