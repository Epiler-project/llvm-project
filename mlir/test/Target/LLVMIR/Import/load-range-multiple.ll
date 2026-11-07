; RUN: not mlir-translate -import-llvm %s 2>&1 | FileCheck %s

; A single ConstantRangeAttr cannot represent a disjoint interval union.
; Reject that valid LLVM shape explicitly instead of dropping/widening it.
; CHECK: load range metadata requires one interval
define i32 @multiple_intervals(ptr %ptr) {
  %value = load i32, ptr %ptr, align 4, !range !0
  ret i32 %value
}
!0 = !{i32 0, i32 16, i32 32, i32 64}
