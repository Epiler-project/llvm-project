; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/argument.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/generic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GENERIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/fixed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIXED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/bundle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BUNDLE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/memcpy.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/offset-poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/offset-range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; BUNDLE: Tensix intrinsic requires a direct C call without operand bundles
; DEFINED: Dst offset must be defined and non-poison
; RANGE: Dst offset must be proven in [0, 1023]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=-xtttensixbh -O0 %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; EXECUTOR: Tensix intrinsic requires a tensix-executor function attribute
; FEATURE: Tensix intrinsic requires +xtttensixbh
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/stateid.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATEID
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/stateid-dynamic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC-STATEID
; STATEID: Tensix SFPU StateID configuration must be static zero
; DYNAMIC-STATEID: Tensix SFPU StateID configuration must be static zero
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/overflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/loop-poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/loop-unknown.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/budget.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/unsigned-overflow-loop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/signed-overflow-loop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/dynamic-overflow-loop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; ARG: SFPU carrier cannot cross the argument ABI
; GENERIC: SFPU carrier is only legal in target intrinsics and PHI
; CONSTANT: SFPU carrier constants require explicit target initialization
; FIXED: immediate operand 0 must be in [0, 7]
; CALL: call has no verified SFPU preservation ABI

;--- argument.ll
define void @test(<32 x i32> %input) "tensix-executor"="trisc1" {
  ret void
}
;--- generic.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
define void @test() "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %b = add <32 x i32> %a, %a
  call void @llvm.riscv.tt.lreg.write(i32 0, <32 x i32> %b)
  ret void
}
;--- poison.ll
declare <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32>, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> poison, i32 0, i32 10)
  ret void
}
;--- fixed.ll
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 8)
  ret void
}
;--- call.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @external_scalar()
define void @test() "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @external_scalar()
  ret void
}

;--- bundle.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpnop()
define void @test() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.sfpnop() [ "deopt"(<32 x i32> %v) ]
  ret void
}
;--- memcpy.ll
declare void @llvm.riscv.tt.sfpnop()
declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1 immarg)
define void @test(ptr %a, ptr %b, i32 %n) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %b, i32 %n, i1 false)
  ret void
}
;--- offset-poison.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %offset = zext i10 poison to i32
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  ret void
}
;--- offset-range.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test(i32 noundef %offset) "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  ret void
}

;--- executor.ll
declare void @llvm.riscv.tt.sfpnop()
define void @test() {
  call void @llvm.riscv.tt.sfpnop()
  ret void
}

;--- stateid.ll
declare void @llvm.riscv.tt.sfpnop()
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16(i32 1, i32 0)
  call void @llvm.riscv.tt.sfpnop()
  ret void
}
;--- stateid-dynamic.ll
declare void @llvm.riscv.tt.sfpnop()
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test(i16 noundef %input) "tensix-executor"="trisc1" {
  %state = zext i16 %input to i32
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 %state, i32 0)
  call void @llvm.riscv.tt.sfpnop()
  ret void
}

;--- overflow.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %wrapped = add nuw i10 1023, 1
  %offset = zext i10 %wrapped to i32
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  ret void
}
;--- loop-poison.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %i = phi i10 [poison, %entry], [%next, %loop]
  %offset = zext i10 %i to i32
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add i10 %i, 1
  %more = icmp ult i10 %next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
;--- loop-unknown.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test(i10 %start) "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %i = phi i10 [%start, %entry], [%next, %loop]
  %offset = zext i10 %i to i32
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add i10 %i, 1
  %more = icmp ult i10 %next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

;--- budget.ll
; A deep scalar dependency graph exceeds the bounded proof budget. The graph
; is straight-line input, not an unrolled loop produced by this compiler.
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test(i32 noundef %seed) "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %n0 = mul i32 %seed, 33
  %n1 = xor i32 %n0, 29
  %n2 = mul i32 %n1, 33
  %n3 = xor i32 %n2, 29
  %n4 = mul i32 %n3, 33
  %n5 = xor i32 %n4, 29
  %n6 = mul i32 %n5, 33
  %n7 = xor i32 %n6, 29
  %n8 = mul i32 %n7, 33
  %n9 = xor i32 %n8, 29
  %n10 = mul i32 %n9, 33
  %n11 = xor i32 %n10, 29
  %n12 = mul i32 %n11, 33
  %n13 = xor i32 %n12, 29
  %n14 = mul i32 %n13, 33
  %n15 = xor i32 %n14, 29
  %n16 = mul i32 %n15, 33
  %n17 = xor i32 %n16, 29
  %n18 = mul i32 %n17, 33
  %n19 = xor i32 %n18, 29
  %n20 = mul i32 %n19, 33
  %n21 = xor i32 %n20, 29
  %n22 = mul i32 %n21, 33
  %n23 = xor i32 %n22, 29
  %n24 = mul i32 %n23, 33
  %n25 = xor i32 %n24, 29
  %n26 = mul i32 %n25, 33
  %n27 = xor i32 %n26, 29
  %n28 = mul i32 %n27, 33
  %n29 = xor i32 %n28, 29
  %n30 = mul i32 %n29, 33
  %n31 = xor i32 %n30, 29
  %n32 = mul i32 %n31, 33
  %n33 = xor i32 %n32, 29
  %n34 = mul i32 %n33, 33
  %n35 = xor i32 %n34, 29
  %n36 = mul i32 %n35, 33
  %n37 = xor i32 %n36, 29
  %n38 = mul i32 %n37, 33
  %n39 = xor i32 %n38, 29
  %n40 = mul i32 %n39, 33
  %n41 = xor i32 %n40, 29
  %n42 = mul i32 %n41, 33
  %n43 = xor i32 %n42, 29
  %n44 = mul i32 %n43, 33
  %n45 = xor i32 %n44, 29
  %n46 = mul i32 %n45, 33
  %n47 = xor i32 %n46, 29
  %n48 = mul i32 %n47, 33
  %n49 = xor i32 %n48, 29
  %n50 = mul i32 %n49, 33
  %n51 = xor i32 %n50, 29
  %n52 = mul i32 %n51, 33
  %n53 = xor i32 %n52, 29
  %n54 = mul i32 %n53, 33
  %n55 = xor i32 %n54, 29
  %n56 = mul i32 %n55, 33
  %n57 = xor i32 %n56, 29
  %n58 = mul i32 %n57, 33
  %n59 = xor i32 %n58, 29
  %n60 = mul i32 %n59, 33
  %n61 = xor i32 %n60, 29
  %n62 = mul i32 %n61, 33
  %n63 = xor i32 %n62, 29
  %n64 = mul i32 %n63, 33
  %n65 = xor i32 %n64, 29
  %n66 = mul i32 %n65, 33
  %n67 = xor i32 %n66, 29
  %n68 = mul i32 %n67, 33
  %n69 = xor i32 %n68, 29
  %n70 = mul i32 %n69, 33
  %n71 = xor i32 %n70, 29
  %n72 = mul i32 %n71, 33
  %n73 = xor i32 %n72, 29
  %n74 = mul i32 %n73, 33
  %n75 = xor i32 %n74, 29
  %n76 = mul i32 %n75, 33
  %n77 = xor i32 %n76, 29
  %n78 = mul i32 %n77, 33
  %n79 = xor i32 %n78, 29
  %n80 = mul i32 %n79, 33
  %n81 = xor i32 %n80, 29
  %n82 = mul i32 %n81, 33
  %n83 = xor i32 %n82, 29
  %n84 = mul i32 %n83, 33
  %n85 = xor i32 %n84, 29
  %n86 = mul i32 %n85, 33
  %n87 = xor i32 %n86, 29
  %n88 = mul i32 %n87, 33
  %n89 = xor i32 %n88, 29
  %n90 = mul i32 %n89, 33
  %n91 = xor i32 %n90, 29
  %n92 = mul i32 %n91, 33
  %n93 = xor i32 %n92, 29
  %n94 = mul i32 %n93, 33
  %n95 = xor i32 %n94, 29
  %n96 = mul i32 %n95, 33
  %n97 = xor i32 %n96, 29
  %n98 = mul i32 %n97, 33
  %n99 = xor i32 %n98, 29
  %n100 = mul i32 %n99, 33
  %n101 = xor i32 %n100, 29
  %n102 = mul i32 %n101, 33
  %n103 = xor i32 %n102, 29
  %n104 = mul i32 %n103, 33
  %n105 = xor i32 %n104, 29
  %n106 = mul i32 %n105, 33
  %n107 = xor i32 %n106, 29
  %n108 = mul i32 %n107, 33
  %n109 = xor i32 %n108, 29
  %n110 = mul i32 %n109, 33
  %n111 = xor i32 %n110, 29
  %n112 = mul i32 %n111, 33
  %n113 = xor i32 %n112, 29
  %n114 = mul i32 %n113, 33
  %n115 = xor i32 %n114, 29
  %n116 = mul i32 %n115, 33
  %n117 = xor i32 %n116, 29
  %n118 = mul i32 %n117, 33
  %n119 = xor i32 %n118, 29
  %n120 = mul i32 %n119, 33
  %n121 = xor i32 %n120, 29
  %n122 = mul i32 %n121, 33
  %n123 = xor i32 %n122, 29
  %n124 = mul i32 %n123, 33
  %n125 = xor i32 %n124, 29
  %n126 = mul i32 %n125, 33
  %n127 = xor i32 %n126, 29
  %n128 = mul i32 %n127, 33
  %n129 = xor i32 %n128, 29
  %n130 = mul i32 %n129, 33
  %n131 = xor i32 %n130, 29
  %n132 = mul i32 %n131, 33
  %n133 = xor i32 %n132, 29
  %n134 = mul i32 %n133, 33
  %n135 = xor i32 %n134, 29
  %n136 = mul i32 %n135, 33
  %n137 = xor i32 %n136, 29
  %n138 = mul i32 %n137, 33
  %n139 = xor i32 %n138, 29
  %n140 = mul i32 %n139, 33
  %n141 = xor i32 %n140, 29
  %n142 = mul i32 %n141, 33
  %n143 = xor i32 %n142, 29
  %n144 = mul i32 %n143, 33
  %n145 = xor i32 %n144, 29
  %n146 = mul i32 %n145, 33
  %n147 = xor i32 %n146, 29
  %n148 = mul i32 %n147, 33
  %n149 = xor i32 %n148, 29
  %n150 = mul i32 %n149, 33
  %n151 = xor i32 %n150, 29
  %n152 = mul i32 %n151, 33
  %n153 = xor i32 %n152, 29
  %n154 = mul i32 %n153, 33
  %n155 = xor i32 %n154, 29
  %n156 = mul i32 %n155, 33
  %n157 = xor i32 %n156, 29
  %n158 = mul i32 %n157, 33
  %n159 = xor i32 %n158, 29
  %offset = and i32 %n159, 1023
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  ret void
}

;--- unsigned-overflow-loop.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %count = phi i32 [0, %entry], [%count.next, %loop]
  %i = phi i32 [4294967290, %entry], [%next, %loop]
  %offset = and i32 %i, 1023
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add nuw i32 %i, 1
  %count.next = add i32 %count, 1
  %more = icmp ult i32 %count.next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

;--- signed-overflow-loop.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  br label %loop
loop:
  %count = phi i32 [0, %entry], [%count.next, %loop]
  %i = phi i32 [2147483642, %entry], [%next, %loop]
  %offset = and i32 %i, 1023
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add nsw i32 %i, 1
  %count.next = add i32 %count, 1
  %more = icmp ult i32 %count.next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

;--- dynamic-overflow-loop.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test(i32 noundef %input) "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %start = or i32 %input, -16
  br label %loop
loop:
  %count = phi i32 [0, %entry], [%count.next, %loop]
  %i = phi i32 [%start, %entry], [%next, %loop]
  %offset = and i32 %i, 1023
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 %offset, i32 0, i32 4)
  %next = add nuw i32 %i, 1
  %count.next = add i32 %count, 1
  %more = icmp ult i32 %count.next, 16
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
