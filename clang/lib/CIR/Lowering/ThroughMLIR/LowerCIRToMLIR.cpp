#include "LowerToMLIRHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {

// class CIRLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
// public:
//   using OpConversionPattern<cir::LoadOp>::OpConversionPattern;

//   mlir::LogicalResult
//   matchAndRewrite(cir::LoadOp op, OpAdaptor adaptor,
//                   mlir::ConversionPatternRewriter &rewriter) const override {
//     mlir::Value base;
//     SmallVector<mlir::Value> indices;
//     SmallVector<mlir::Operation *> eraseList;
//     mlir::memref::LoadOp newLoad;
//     if (findBaseAndIndices(adaptor.getAddr(), base, indices, eraseList,
//                            rewriter)) {
//       newLoad = rewriter.create<mlir::memref::LoadOp>(
//           op.getLoc(), base, indices, op.getIsNontemporal());
//       eraseIfSafe(op.getAddr(), base, eraseList, rewriter);
//     } else
//       newLoad = rewriter.create<mlir::memref::LoadOp>(
//           op.getLoc(), adaptor.getAddr(), mlir::ValueRange{},
//           op.getIsNontemporal());

//     // Convert adapted result to its original type if needed.
//     mlir::Value result = emitFromMemory(rewriter, op, newLoad.getResult());
//     rewriter.replaceOp(op, result);
//     return mlir::LogicalResult::success();
//   }
// };

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect,
                    mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the CIR dialect module to MLIR standard dialects";
  }

  StringRef getArgument() const override { return "cir-to-mlir"; }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  // patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  // patterns
  //     .add<CIRATanOpLowering, CIRCmpOpLowering, CIRCallOpLowering,
  //          CIRUnaryOpLowering, CIRBinOpLowering, CIRLoadOpLowering,
  //          CIRConstantOpLowering, CIRStoreOpLowering, CIRAllocaOpLowering,
  //          CIRFuncOpLowering, CIRScopeOpLowering, CIRBrCondOpLowering,
  //          CIRTernaryOpLowering, CIRYieldOpLowering, CIRCosOpLowering,
  //          CIRGlobalOpLowering, CIRGetGlobalOpLowering, CIRCastOpLowering,
  //          CIRPtrStrideOpLowering, CIRGetElementOpLowering,
  //          CIRSqrtOpLowering, CIRCeilOpLowering, CIRExp2OpLowering,
  //          CIRExpOpLowering, CIRFAbsOpLowering, CIRAbsOpLowering,
  //          CIRFloorOpLowering, CIRLog10OpLowering, CIRLog2OpLowering,
  //          CIRLogOpLowering, CIRRoundOpLowering, CIRPtrStrideOpLowering,
  //          CIRSinOpLowering, CIRShiftOpLowering, CIRBitClzOpLowering,
  //          CIRBitCtzOpLowering, CIRBitPopcountOpLowering,
  //          CIRBitClrsbOpLowering, CIRBitFfsOpLowering,
  //          CIRBitParityOpLowering, CIRIfOpLowering, CIRVectorCreateLowering,
  //          CIRVectorInsertLowering, CIRVectorExtractLowering,
  //          CIRVectorCmpOpLowering, CIRACosOpLowering, CIRASinOpLowering,
  //          CIRUnreachableOpLowering, CIRTanOpLowering, CIRTrapOpLowering>(
  //         converter, patterns.getContext());
  // patterns.add<CIRLoadOpLowering>(converter, patterns.getContext());
}

} // namespace cir