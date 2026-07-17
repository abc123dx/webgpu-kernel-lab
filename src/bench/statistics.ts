export interface BenchmarkStatistics {
  iterations: number;
  minimumMs: number;
  maximumMs: number;
  meanMs: number;
  medianMs: number;
  p95Ms: number;
  standardDeviationMs: number;
}

export function summarizeDurations(samples: readonly number[]): BenchmarkStatistics {
  if (samples.length === 0) {
    throw new RangeError("至少需要一个耗时样本");
  }
  if (samples.some((sample) => !Number.isFinite(sample) || sample < 0)) {
    throw new RangeError("耗时样本必须是有限的非负数");
  }

  const sorted = [...samples].sort((left, right) => left - right);
  const mean = sorted.reduce((total, sample) => total + sample, 0) / sorted.length;
  const variance =
    sorted.reduce((total, sample) => total + (sample - mean) ** 2, 0) /
    sorted.length;

  return {
    iterations: sorted.length,
    minimumMs: sorted[0] ?? 0,
    maximumMs: sorted.at(-1) ?? 0,
    meanMs: mean,
    medianMs: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    standardDeviationMs: Math.sqrt(variance),
  };
}

export function percentile(sortedSamples: readonly number[], quantile: number): number {
  if (sortedSamples.length === 0) {
    throw new RangeError("至少需要一个已排序样本");
  }
  if (quantile < 0 || quantile > 1 || !Number.isFinite(quantile)) {
    throw new RangeError("分位数必须介于 0 和 1 之间");
  }
  const position = (sortedSamples.length - 1) * quantile;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const lower = sortedSamples[lowerIndex] ?? 0;
  const upper = sortedSamples[upperIndex] ?? lower;
  return lower + (upper - lower) * (position - lowerIndex);
}

export function formatMilliseconds(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "—";
  }
  if (value < 0.01) {
    return "<0.01 ms";
  }
  if (value < 10) {
    return `${value.toFixed(2)} ms`;
  }
  return `${value.toFixed(1)} ms`;
}

export function calculateThroughput(
  operationCount: number,
  elapsedMs: number,
): number {
  if (operationCount < 0 || !Number.isFinite(operationCount)) {
    throw new RangeError("运算次数必须是有限的非负数");
  }
  if (elapsedMs <= 0 || !Number.isFinite(elapsedMs)) {
    return 0;
  }
  return operationCount / (elapsedMs / 1000);
}
