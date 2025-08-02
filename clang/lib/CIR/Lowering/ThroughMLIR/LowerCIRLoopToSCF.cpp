#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class SCFLoop {
public:
  SCFLoop(cir::ForOp op, mlir::ConversionPatternRewriter *rewriter)
      : forOp(op), rewriter(rewriter) {}

  int64_t getStep() { return step; }
  mlir::Value getLowerBound() { return lowerBound; }
  mlir::Value getUpperBound() { return upperBound; }
  bool isCanonical() { return canonical; }
  bool hasBreakOrContinue() { return hasBreakContinue; }

  // Returns true if successfully finds both step and induction variable.
  mlir::LogicalResult findStepAndIV();
  cir::CmpOp findCmpOp();
  mlir::Value findIVInitValue();
  void analysis();

  mlir::Value plusConstant(mlir::Value v, mlir::Location loc, int addend);
  void transferToSCFForOp();
  void transformToSCFWhileOp();
  void transformToCIRWhileOp(); // TODO

private:
  cir::ForOp forOp;
  cir::CmpOp cmpOp;
  mlir::Value ivAddr, lowerBound = nullptr, upperBound = nullptr;
  mlir::ConversionPatternRewriter *rewriter;
  int64_t step = 0;
  bool hasBreakContinue = false;
  bool canonical = true;
};

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter) {
  // patterns.add<CIRForOpLowering, CIRWhileOpLowering, CIRConditionOpLowering,
  //              CIRDoOpLowering>(converter, patterns.getContext());
}

} // namespace cir
