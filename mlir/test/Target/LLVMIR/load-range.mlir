// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s
// RUN: mlir-translate -mlir-to-llvmir %s | mlir-translate -import-llvm | FileCheck %s --check-prefix=ROUNDTRIP

// CHECK-LABEL: define i32 @bounded_word
// CHECK: load i32, ptr %{{.*}}, align 4, !range ![[RANGE:[0-9]+]], !noundef ![[DEFINED:[0-9]+]]
// ROUNDTRIP-LABEL: llvm.func @bounded_word
// ROUNDTRIP: llvm.load {{.*}}<{{.*}}range = <i32, 0, 64>, noundef>
llvm.func @bounded_word(%ptr: !llvm.ptr) -> i32 {
  %value = llvm.load %ptr <alignment = 4, range = #llvm.constant_range<i32, 0, 64>, noundef = unit> : !llvm.ptr -> i32
  llvm.return %value : i32
}

// CHECK-LABEL: define i8 @wrapping_word
// CHECK: load i8, ptr %{{.*}}, align 1, !range ![[WRAP:[0-9]+]]
// ROUNDTRIP-LABEL: llvm.func @wrapping_word
// ROUNDTRIP: llvm.load {{.*}}<{{.*}}range = <i8, -16, 16>>
llvm.func @wrapping_word(%ptr: !llvm.ptr) -> i8 {
  %value = llvm.load %ptr <alignment = 1, range = #llvm.constant_range<i8, -16, 16>> : !llvm.ptr -> i8
  llvm.return %value : i8
}

// A noundef contract is independent of an integer range.
// CHECK-LABEL: define float @defined_float
// CHECK: load float, ptr %{{.*}}, align 4, !noundef ![[DEFINED]]
// ROUNDTRIP-LABEL: llvm.func @defined_float
// ROUNDTRIP: llvm.load {{.*}}<{{.*}}noundef>
llvm.func @defined_float(%ptr: !llvm.ptr) -> f32 {
  %value = llvm.load %ptr <alignment = 4, noundef = unit> : !llvm.ptr -> f32
  llvm.return %value : f32
}

// CHECK-LABEL: define i32 @ordinary_word
// CHECK: load i32, ptr %{{.*}}, align 4{{$}}
// ROUNDTRIP-LABEL: llvm.func @ordinary_word
// ROUNDTRIP: llvm.load {{.*}}<alignment = 4> : !llvm.ptr -> i32
llvm.func @ordinary_word(%ptr: !llvm.ptr) -> i32 {
  %value = llvm.load %ptr <alignment = 4> : !llvm.ptr -> i32
  llvm.return %value : i32
}

// CHECK-LABEL: define <4 x i8> @bounded_vector
// CHECK: load <4 x i8>, ptr %{{.*}}, align 4, !range ![[VECTOR:[0-9]+]], !noundef ![[DEFINED]]
// ROUNDTRIP-LABEL: llvm.func @bounded_vector
// ROUNDTRIP: llvm.load {{.*}}<{{.*}}range = <i8, 0, 16>, noundef>
llvm.func @bounded_vector(%ptr: !llvm.ptr) -> vector<4xi8> {
  %value = llvm.load %ptr <alignment = 4, range = #llvm.constant_range<i8, 0, 16>, noundef = unit> : !llvm.ptr -> vector<4xi8>
  llvm.return %value : vector<4xi8>
}

// CHECK-DAG: ![[RANGE]] = !{i32 0, i32 64}
// CHECK-DAG: ![[DEFINED]] = !{}
// CHECK-DAG: ![[WRAP]] = !{i8 -16, i8 16}
// CHECK-DAG: ![[VECTOR]] = !{i8 0, i8 16}
