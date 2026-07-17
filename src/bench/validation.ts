export interface ValidationResult {
  passed: boolean;
  tolerance: number;
  comparedValues: number;
  mismatchCount: number;
  maximumAbsoluteError: number;
  meanAbsoluteError: number;
  rootMeanSquareError: number;
}

export function compareFloatArrays(
  actual: Float32Array,
  expected: Float32Array,
  tolerance = 1e-4,
): ValidationResult {
  if (actual.length !== expected.length) {
    throw new RangeError("实际数组与期望数组必须长度相同");
  }
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError("误差容限必须是有限的非负数");
  }

  let maximumAbsoluteError = 0;
  let absoluteErrorTotal = 0;
  let squaredErrorTotal = 0;
  let mismatchCount = 0;

  for (let index = 0; index < actual.length; index += 1) {
    const error = Math.abs((actual[index] ?? 0) - (expected[index] ?? 0));
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    absoluteErrorTotal += error;
    squaredErrorTotal += error ** 2;
    if (error > tolerance || !Number.isFinite(error)) {
      mismatchCount += 1;
    }
  }

  const denominator = Math.max(1, actual.length);
  return {
    passed: mismatchCount === 0,
    tolerance,
    comparedValues: actual.length,
    mismatchCount,
    maximumAbsoluteError,
    meanAbsoluteError: absoluteErrorTotal / denominator,
    rootMeanSquareError: Math.sqrt(squaredErrorTotal / denominator),
  };
}
