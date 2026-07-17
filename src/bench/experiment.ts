import {
  CONVOLUTION_FILTERS,
  createMatrix,
  createSequence,
  createTestImage,
} from "../kernels/data";
import { imageConvolutionCpu } from "../kernels/image-convolution";
import { matrixMultiplyCpu } from "../kernels/matrix-multiply";
import type {
  ConvolutionPreset,
  KernelKind,
} from "../kernels/types";
import { vectorAddCpu } from "../kernels/vector-add";
import type {
  GpuExecutionResult,
  WebGpuRuntime,
} from "../runtime/webgpu";
import {
  calculateThroughput,
  summarizeDurations,
  type BenchmarkStatistics,
} from "./statistics";
import {
  compareFloatArrays,
  type ValidationResult,
} from "./validation";

export type BackendPreference = "auto" | "webgpu" | "cpu";
export type ActualBackend = "webgpu" | "cpu";
export type ExperimentPhase = "preparing" | "warming" | "benchmarking" | "validating";

export interface ExperimentConfig {
  kernel: KernelKind;
  backend: BackendPreference;
  vectorLength: number;
  matrixSize: number;
  imageSize: number;
  convolutionPreset: ConvolutionPreset;
  warmupIterations: number;
  benchmarkIterations: number;
}

export interface ExperimentResult {
  schemaVersion: "1.0";
  createdAt: string;
  kernel: KernelKind;
  requestedBackend: BackendPreference;
  backend: ActualBackend;
  fallbackReason: string | null;
  parameters: Record<string, number | string>;
  dispatch: readonly [number, number, number] | null;
  benchmark: BenchmarkStatistics;
  cpuReference: BenchmarkStatistics;
  validation: ValidationResult;
  operationCount: number;
  operationsPerSecond: number;
  inputBytes: number;
  outputBytes: number;
  outputPreview: number[];
}

type ProgressCallback = (phase: ExperimentPhase, completed: number, total: number) => void;

interface PreparedExperiment {
  parameters: Record<string, number | string>;
  inputBytes: number;
  outputBytes: number;
  operationCount: number;
  cpu: () => Float32Array;
  gpu: (runtime: WebGpuRuntime) => Promise<GpuExecutionResult>;
}

export async function runExperiment(
  config: ExperimentConfig,
  runtime: WebGpuRuntime | null,
  onProgress?: ProgressCallback,
): Promise<ExperimentResult> {
  validateConfig(config);
  onProgress?.("preparing", 0, 1);
  const prepared = prepareExperiment(config);
  const shouldUseGpu = config.backend !== "cpu" && runtime !== null;
  const fallbackReason =
    config.backend !== "cpu" && !runtime
      ? "WebGPU was unavailable, so the CPU reference path was used."
      : null;

  const cpuRun = await benchmarkCpu(
    prepared.cpu,
    config.warmupIterations,
    config.benchmarkIterations,
    onProgress,
  );

  let backend: ActualBackend = "cpu";
  let output = cpuRun.output;
  let dispatch: readonly [number, number, number] | null = null;
  let benchmark = cpuRun.statistics;

  if (shouldUseGpu && runtime) {
    backend = "webgpu";
    const gpuRun = await benchmarkGpu(
      () => prepared.gpu(runtime),
      config.warmupIterations,
      config.benchmarkIterations,
      onProgress,
    );
    output = gpuRun.output;
    dispatch = gpuRun.dispatch;
    benchmark = gpuRun.statistics;
  }

  onProgress?.("validating", 1, 1);
  const validation = compareFloatArrays(output, cpuRun.output, toleranceFor(config.kernel));

  return {
    schemaVersion: "1.0",
    createdAt: new Date().toISOString(),
    kernel: config.kernel,
    requestedBackend: config.backend,
    backend,
    fallbackReason,
    parameters: prepared.parameters,
    dispatch,
    benchmark,
    cpuReference: cpuRun.statistics,
    validation,
    operationCount: prepared.operationCount,
    operationsPerSecond: calculateThroughput(
      prepared.operationCount,
      benchmark.medianMs,
    ),
    inputBytes: prepared.inputBytes,
    outputBytes: prepared.outputBytes,
    outputPreview: Array.from(output.slice(0, 12), roundForExport),
  };
}

export function validateConfig(config: ExperimentConfig): void {
  assertIntegerInRange(config.warmupIterations, 0, 10, "warmup iterations");
  assertIntegerInRange(config.benchmarkIterations, 1, 30, "benchmark iterations");
  assertIntegerInRange(config.vectorLength, 64, 4_194_304, "vector length");
  assertIntegerInRange(config.matrixSize, 2, 192, "matrix size");
  assertIntegerInRange(config.imageSize, 8, 1024, "image size");
}

export function operationCountFor(
  kernel: KernelKind,
  size: number,
): number {
  switch (kernel) {
    case "vector-add":
      return size;
    case "matrix-multiply":
      return 2 * size ** 3;
    case "image-convolution":
      return size * size * 3 * 9 * 2;
  }
}

function prepareExperiment(config: ExperimentConfig): PreparedExperiment {
  switch (config.kernel) {
    case "vector-add": {
      const inputA = createSequence(config.vectorLength, 0xa11ce);
      const inputB = createSequence(config.vectorLength, 0xbeef);
      return {
        parameters: { length: config.vectorLength },
        inputBytes: inputA.byteLength + inputB.byteLength,
        outputBytes: inputA.byteLength,
        operationCount: operationCountFor(config.kernel, config.vectorLength),
        cpu: () => vectorAddCpu(inputA, inputB),
        gpu: (runtime) => runtime.runVectorAdd(inputA, inputB),
      };
    }
    case "matrix-multiply": {
      const matrixA = createMatrix(config.matrixSize, 0xa11ce);
      const matrixB = createMatrix(config.matrixSize, 0xbeef);
      return {
        parameters: { size: config.matrixSize },
        inputBytes: matrixA.byteLength + matrixB.byteLength,
        outputBytes: matrixA.byteLength,
        operationCount: operationCountFor(config.kernel, config.matrixSize),
        cpu: () => matrixMultiplyCpu(matrixA, matrixB, config.matrixSize),
        gpu: (runtime) =>
          runtime.runMatrixMultiply(matrixA, matrixB, config.matrixSize),
      };
    }
    case "image-convolution": {
      const image = createTestImage(config.imageSize);
      const filter = CONVOLUTION_FILTERS[config.convolutionPreset];
      return {
        parameters: {
          width: config.imageSize,
          height: config.imageSize,
          filter: config.convolutionPreset,
        },
        inputBytes: image.byteLength + filter.byteLength,
        outputBytes: image.byteLength,
        operationCount: operationCountFor(config.kernel, config.imageSize),
        cpu: () =>
          imageConvolutionCpu(
            image,
            config.imageSize,
            config.imageSize,
            filter,
          ),
        gpu: (runtime) =>
          runtime.runImageConvolution(
            image,
            config.imageSize,
            config.imageSize,
            filter,
          ),
      };
    }
  }
}

async function benchmarkCpu(
  task: () => Float32Array,
  warmups: number,
  iterations: number,
  onProgress?: ProgressCallback,
): Promise<{ output: Float32Array; statistics: BenchmarkStatistics }> {
  onProgress?.("warming", 0, warmups);
  for (let index = 0; index < warmups; index += 1) {
    task();
    onProgress?.("warming", index + 1, warmups);
    await yieldToBrowser();
  }

  const durations: number[] = [];
  let output: Float32Array<ArrayBufferLike> = new Float32Array();
  for (let index = 0; index < iterations; index += 1) {
    const startedAt = performance.now();
    output = task();
    durations.push(performance.now() - startedAt);
    onProgress?.("benchmarking", index + 1, iterations);
    await yieldToBrowser();
  }
  return { output, statistics: summarizeDurations(durations) };
}

async function benchmarkGpu(
  task: () => Promise<GpuExecutionResult>,
  warmups: number,
  iterations: number,
  onProgress?: ProgressCallback,
): Promise<{
  output: Float32Array;
  dispatch: readonly [number, number, number];
  statistics: BenchmarkStatistics;
}> {
  onProgress?.("warming", 0, warmups);
  for (let index = 0; index < warmups; index += 1) {
    await task();
    onProgress?.("warming", index + 1, warmups);
  }

  const durations: number[] = [];
  let lastResult: GpuExecutionResult | null = null;
  for (let index = 0; index < iterations; index += 1) {
    lastResult = await task();
    durations.push(lastResult.elapsedMs);
    onProgress?.("benchmarking", index + 1, iterations);
  }
  if (!lastResult) {
    throw new Error("GPU benchmark produced no result");
  }
  return {
    output: lastResult.values,
    dispatch: lastResult.dispatch,
    statistics: summarizeDurations(durations),
  };
}

function assertIntegerInRange(
  value: number,
  minimum: number,
  maximum: number,
  label: string,
): void {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${label} must be an integer between ${minimum} and ${maximum}`);
  }
}

function toleranceFor(kind: KernelKind): number {
  return kind === "matrix-multiply" ? 5e-4 : 1e-5;
}

function roundForExport(value: number): number {
  return Number(value.toFixed(6));
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });
}
