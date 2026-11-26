; Native-width Zaamo RMW operations do not require Zalrsc.
; This covers the native AMO contract used by TReX. It does not establish
; general Zaamo-only legalization for cmpxchg, subword RMW or unsupported ops.
;
; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+zaamo,-zalrsc -verify-machineinstrs < %t/word.ll | FileCheck %s --check-prefixes=WORD,WMO-WORD --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc -verify-machineinstrs < %t/word.ll | FileCheck %s --check-prefixes=WORD,WMO-WORD --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv32 -mattr=+zaamo,-zalrsc,+ztso -verify-machineinstrs < %t/word.ll | FileCheck %s --check-prefixes=WORD,TSO-WORD --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc,+ztso -verify-machineinstrs < %t/word.ll | FileCheck %s --check-prefixes=WORD,TSO-WORD --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc -verify-machineinstrs < %t/double.ll | FileCheck %s --check-prefixes=DOUBLE,WMO-DOUBLE --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.
; RUN: llc -mtriple=riscv64 -mattr=+zaamo,-zalrsc,+ztso -verify-machineinstrs < %t/double.ll | FileCheck %s --check-prefixes=DOUBLE,TSO-DOUBLE --implicit-check-not=__atomic --implicit-check-not=lr. --implicit-check-not=sc.

;--- word.ll

; WORD-LABEL: xchg_i32:
; WORD: amoswap.w{{[ \t]}}
; WORD: ret
define i32 @xchg_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: add_i32:
; WORD: amoadd.w{{[ \t]}}
; WORD: ret
define i32 @add_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw add ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: sub_i32:
; WORD: neg {{[a-z0-9]+}}, a1
; WORD: amoadd.w{{[ \t]}}
; WORD: ret
define i32 @sub_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw sub ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: and_i32:
; WORD: amoand.w{{[ \t]}}
; WORD: ret
define i32 @and_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw and ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: or_i32:
; WORD: amoor.w{{[ \t]}}
; WORD: ret
define i32 @or_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw or ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: xor_i32:
; WORD: amoxor.w{{[ \t]}}
; WORD: ret
define i32 @xor_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xor ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: min_i32:
; WORD: amomin.w{{[ \t]}}
; WORD: ret
define i32 @min_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw min ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: max_i32:
; WORD: amomax.w{{[ \t]}}
; WORD: ret
define i32 @max_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw max ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: umin_i32:
; WORD: amominu.w{{[ \t]}}
; WORD: ret
define i32 @umin_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw umin ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: umax_i32:
; WORD: amomaxu.w{{[ \t]}}
; WORD: ret
define i32 @umax_i32(ptr %address, i32 %value) nounwind {
  %old = atomicrmw umax ptr %address, i32 %value monotonic, align 4
  ret i32 %old
}

; WORD-LABEL: xchg_i32_acquire:
; WMO-WORD: amoswap.w.aq{{[ \t]}}
; TSO-WORD: amoswap.w{{[ \t]}}
; WORD: ret
define i32 @xchg_i32_acquire(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i32 %value acquire, align 4
  ret i32 %old
}

; WORD-LABEL: xchg_i32_release:
; WMO-WORD: amoswap.w.rl{{[ \t]}}
; TSO-WORD: amoswap.w{{[ \t]}}
; WORD: ret
define i32 @xchg_i32_release(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i32 %value release, align 4
  ret i32 %old
}

; WORD-LABEL: xchg_i32_acq_rel:
; WMO-WORD: amoswap.w.aqrl{{[ \t]}}
; TSO-WORD: amoswap.w{{[ \t]}}
; WORD: ret
define i32 @xchg_i32_acq_rel(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i32 %value acq_rel, align 4
  ret i32 %old
}

; WORD-LABEL: xchg_i32_seq_cst:
; WMO-WORD: amoswap.w.aqrl{{[ \t]}}
; TSO-WORD: amoswap.w{{[ \t]}}
; WORD: ret
define i32 @xchg_i32_seq_cst(ptr %address, i32 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i32 %value seq_cst, align 4
  ret i32 %old
}

; WORD-LABEL: add_i32_unused:
; WMO-WORD: amoadd.w.aqrl zero,
; TSO-WORD: amoadd.w zero,
; WORD: ret
define void @add_i32_unused(ptr %address, i32 %value) nounwind {
  %old = atomicrmw add ptr %address, i32 %value seq_cst, align 4
  ret void
}
;--- double.ll

; DOUBLE-LABEL: xchg_i64:
; DOUBLE: amoswap.d{{[ \t]}}
; DOUBLE: ret
define i64 @xchg_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: add_i64:
; DOUBLE: amoadd.d{{[ \t]}}
; DOUBLE: ret
define i64 @add_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw add ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: sub_i64:
; DOUBLE: neg {{[a-z0-9]+}}, a1
; DOUBLE: amoadd.d{{[ \t]}}
; DOUBLE: ret
define i64 @sub_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw sub ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: and_i64:
; DOUBLE: amoand.d{{[ \t]}}
; DOUBLE: ret
define i64 @and_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw and ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: or_i64:
; DOUBLE: amoor.d{{[ \t]}}
; DOUBLE: ret
define i64 @or_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw or ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: xor_i64:
; DOUBLE: amoxor.d{{[ \t]}}
; DOUBLE: ret
define i64 @xor_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xor ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: min_i64:
; DOUBLE: amomin.d{{[ \t]}}
; DOUBLE: ret
define i64 @min_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw min ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: max_i64:
; DOUBLE: amomax.d{{[ \t]}}
; DOUBLE: ret
define i64 @max_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw max ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: umin_i64:
; DOUBLE: amominu.d{{[ \t]}}
; DOUBLE: ret
define i64 @umin_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw umin ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: umax_i64:
; DOUBLE: amomaxu.d{{[ \t]}}
; DOUBLE: ret
define i64 @umax_i64(ptr %address, i64 %value) nounwind {
  %old = atomicrmw umax ptr %address, i64 %value monotonic, align 8
  ret i64 %old
}

; DOUBLE-LABEL: xchg_i64_acquire:
; WMO-DOUBLE: amoswap.d.aq{{[ \t]}}
; TSO-DOUBLE: amoswap.d{{[ \t]}}
; DOUBLE: ret
define i64 @xchg_i64_acquire(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i64 %value acquire, align 8
  ret i64 %old
}

; DOUBLE-LABEL: xchg_i64_release:
; WMO-DOUBLE: amoswap.d.rl{{[ \t]}}
; TSO-DOUBLE: amoswap.d{{[ \t]}}
; DOUBLE: ret
define i64 @xchg_i64_release(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i64 %value release, align 8
  ret i64 %old
}

; DOUBLE-LABEL: xchg_i64_acq_rel:
; WMO-DOUBLE: amoswap.d.aqrl{{[ \t]}}
; TSO-DOUBLE: amoswap.d{{[ \t]}}
; DOUBLE: ret
define i64 @xchg_i64_acq_rel(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i64 %value acq_rel, align 8
  ret i64 %old
}

; DOUBLE-LABEL: xchg_i64_seq_cst:
; WMO-DOUBLE: amoswap.d.aqrl{{[ \t]}}
; TSO-DOUBLE: amoswap.d{{[ \t]}}
; DOUBLE: ret
define i64 @xchg_i64_seq_cst(ptr %address, i64 %value) nounwind {
  %old = atomicrmw xchg ptr %address, i64 %value seq_cst, align 8
  ret i64 %old
}

; DOUBLE-LABEL: add_i64_unused:
; WMO-DOUBLE: amoadd.d.aqrl zero,
; TSO-DOUBLE: amoadd.d zero,
; DOUBLE: ret
define void @add_i64_unused(ptr %address, i64 %value) nounwind {
  %old = atomicrmw add ptr %address, i64 %value seq_cst, align 8
  ret void
}
