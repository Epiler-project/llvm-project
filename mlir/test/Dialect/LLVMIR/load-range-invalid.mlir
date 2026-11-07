// RUN: mlir-opt -split-input-file -verify-diagnostics %s

llvm.func @floating_range(%ptr: !llvm.ptr) -> f32 {
  // expected-error@+1 {{range requires an integer or integer-vector result}}
  %value = llvm.load %ptr <range = #llvm.constant_range<i32, 0, 64>> : !llvm.ptr -> f32
  llvm.return %value : f32
}

// -----

llvm.func @pointer_range(%ptr: !llvm.ptr) -> !llvm.ptr {
  // expected-error@+1 {{range requires an integer or integer-vector result}}
  %value = llvm.load %ptr <range = #llvm.constant_range<i32, 0, 64>> : !llvm.ptr -> !llvm.ptr
  llvm.return %value : !llvm.ptr
}

// -----

llvm.func @width_mismatch(%ptr: !llvm.ptr) -> i32 {
  // expected-error@+1 {{range bounds must match the result element bit width}}
  %value = llvm.load %ptr <range = #llvm.constant_range<i64, 0, 64>> : !llvm.ptr -> i32
  llvm.return %value : i32
}

// -----

llvm.func @full_range(%ptr: !llvm.ptr) -> i32 {
  // expected-error@+1 {{range must not be empty or full}}
  %value = llvm.load %ptr <range = #llvm.constant_range<i32, 0, 0>> : !llvm.ptr -> i32
  llvm.return %value : i32
}

// -----

llvm.func @empty_range(%ptr: !llvm.ptr) -> i32 {
  // expected-error@+1 {{range must not be empty or full}}
  %value = llvm.load %ptr <range = #llvm.constant_range<i32, -1, -1>> : !llvm.ptr -> i32
  llvm.return %value : i32
}
