; RUN: mlir-translate -import-llvm %s | FileCheck %s
; RUN: mlir-translate -import-llvm %s | mlir-translate -mlir-to-llvmir | FileCheck %s --check-prefix=ROUNDTRIP

; CHECK-LABEL: llvm.func @bounded_word
; CHECK: llvm.load {{.*}}<{{.*}}range = <i32, 0, 64>, noundef>
; ROUNDTRIP-LABEL: define i32 @bounded_word
; ROUNDTRIP: load i32, ptr %{{.*}}, align 4, !range ![[RANGE:[0-9]+]], !noundef ![[DEFINED:[0-9]+]]
define i32 @bounded_word(ptr %ptr) {
  %value = load i32, ptr %ptr, align 4, !range !0, !noundef !1
  ret i32 %value
}

; CHECK-LABEL: llvm.func @wrapping_word
; CHECK: llvm.load {{.*}}<{{.*}}range = <i8, -16, 16>>
; ROUNDTRIP-LABEL: define i8 @wrapping_word
; ROUNDTRIP: load i8, ptr %{{.*}}, align 1, !range ![[WRAP:[0-9]+]]
define i8 @wrapping_word(ptr %ptr) {
  %value = load i8, ptr %ptr, align 1, !range !2
  ret i8 %value
}

; CHECK-LABEL: llvm.func @defined_float
; CHECK: llvm.load {{.*}}<{{.*}}noundef>
; ROUNDTRIP-LABEL: define float @defined_float
; ROUNDTRIP: load float, ptr %{{.*}}, align 4, !noundef ![[DEFINED]]
define float @defined_float(ptr %ptr) {
  %value = load float, ptr %ptr, align 4, !noundef !1
  ret float %value
}

; CHECK-LABEL: llvm.func @bounded_vector
; CHECK: llvm.load {{.*}}<{{.*}}range = <i8, 0, 16>, noundef>
; ROUNDTRIP-LABEL: define <4 x i8> @bounded_vector
; ROUNDTRIP: load <4 x i8>, ptr %{{.*}}, align 4, !range ![[VECTOR:[0-9]+]], !noundef ![[DEFINED]]
define <4 x i8> @bounded_vector(ptr %ptr) {
  %value = load <4 x i8>, ptr %ptr, align 4, !range !3, !noundef !1
  ret <4 x i8> %value
}

; ROUNDTRIP-DAG: ![[RANGE]] = !{i32 0, i32 64}
; ROUNDTRIP-DAG: ![[DEFINED]] = !{}
; ROUNDTRIP-DAG: ![[WRAP]] = !{i8 -16, i8 16}
!0 = !{i32 0, i32 64}
!1 = !{}
!2 = !{i8 -16, i8 16}
; ROUNDTRIP-DAG: ![[VECTOR]] = !{i8 0, i8 16}
!3 = !{i8 0, i8 16}
