; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=call --implicit-check-not=tail
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not=call --implicit-check-not=tail
;
; Address-guard selects can become scalar min/max in the codegen pipeline.
; All received widths must reach real scalar instructions without helper calls.
declare i64 @llvm.umin.i64(i64, i64)
declare i64 @llvm.umax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)
declare i64 @llvm.smax.i64(i64, i64)
declare i32 @llvm.umin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.smax.i32(i32, i32)
declare i64 @llvm.readcyclecounter()

; CHECK-LABEL: integer_bounds_i64:
; CHECK: sd
; CHECK: sd
; CHECK: sd
; CHECK: sd
; CHECK: ret
define void @integer_bounds_i64(i64 %a, i64 %b, i64 %c, i64 %d, ptr %out)
    "trex-bosc-bound-rvv"="1" {
  %umin = call i64 @llvm.umin.i64(i64 %a, i64 %b)
  %umax = call i64 @llvm.umax.i64(i64 %a, i64 %b)
  %smin = call i64 @llvm.smin.i64(i64 %c, i64 %d)
  %smax = call i64 @llvm.smax.i64(i64 %c, i64 %d)
  %p1 = getelementptr i64, ptr %out, i64 1
  %p2 = getelementptr i64, ptr %out, i64 2
  %p3 = getelementptr i64, ptr %out, i64 3
  store volatile i64 %umin, ptr %out
  store volatile i64 %umax, ptr %p1
  store volatile i64 %smin, ptr %p2
  store volatile i64 %smax, ptr %p3
  ret void
}

; CHECK-LABEL: integer_bounds_i32:
; CHECK: sw
; CHECK: sw
; CHECK: sw
; CHECK: sw
; CHECK: ret
define void @integer_bounds_i32(i32 %a, i32 %b, i32 %c, i32 %d, ptr %out)
    "trex-bosc-bound-rvv"="1" {
  %umin = call i32 @llvm.umin.i32(i32 %a, i32 %b)
  %umax = call i32 @llvm.umax.i32(i32 %a, i32 %b)
  %smin = call i32 @llvm.smin.i32(i32 %c, i32 %d)
  %smax = call i32 @llvm.smax.i32(i32 %c, i32 %d)
  %p1 = getelementptr i32, ptr %out, i64 1
  %p2 = getelementptr i32, ptr %out, i64 2
  %p3 = getelementptr i32, ptr %out, i64 3
  store volatile i32 %umin, ptr %out
  store volatile i32 %umax, ptr %p1
  store volatile i32 %smin, ptr %p2
  store volatile i32 %smax, ptr %p3
  ret void
}

; Scalar observation is valid without any bound RVV attribute or setvl.
; CHECK-LABEL: scalar_observation:
; CHECK: rdcycle
; CHECK: ret
define i64 @scalar_observation() {
  %cycles = call i64 @llvm.readcyclecounter()
  ret i64 %cycles
}

; CHECK-LABEL: bound_observation:
; CHECK: rdcycle
; CHECK: ret
define i64 @bound_observation() "trex-bosc-bound-rvv"="1" {
  %cycles = call i64 @llvm.readcyclecounter()
  ret i64 %cycles
}
