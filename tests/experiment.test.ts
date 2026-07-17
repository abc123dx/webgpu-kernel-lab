import { describe, expect, it } from "vitest";
import {
  operationCountFor,
  runExperiment,
  validateConfig,
  type ExperimentConfig,
} from "../src/bench/experiment";
import { createExportDocument, serializeResult } from "../src/bench/export";

const baseConfig: ExperimentConfig = {
  kernel: "vector-add",
  backend: "cpu",
  vectorLength: 65_536,
  matrixSize: 16,
  imageSize: 32,
  convolutionPreset: "gaussian-blur",
  warmupIterations: 0,
  benchmarkIterations: 1,
};

describe("experiment orchestration", () => {
  it("accepts a valid configuration", () => {
    expect(() => validateConfig(baseConfig)).not.toThrow();
  });

  it("rejects excessive benchmark iterations", () => {
    expect(() =>
      validateConfig({ ...baseConfig, benchmarkIterations: 31 }),
    ).toThrow(RangeError);
  });

  it("models operation counts by kernel", () => {
    expect(operationCountFor("vector-add", 10)).toBe(10);
    expect(operationCountFor("matrix-multiply", 10)).toBe(2_000);
    expect(operationCountFor("image-convolution", 10)).toBe(5_400);
  });

  it("runs a CPU vector experiment and validates it", async () => {
    const result = await runExperiment(baseConfig, null);
    expect(result.backend).toBe("cpu");
    expect(result.validation.passed).toBe(true);
    expect(result.outputPreview).toHaveLength(12);
    expect(result.fallbackReason).toBeNull();
  });

  it("records a friendly fallback when auto mode lacks WebGPU", async () => {
    const result = await runExperiment(
      { ...baseConfig, backend: "auto" },
      null,
    );
    expect(result.backend).toBe("cpu");
    expect(result.fallbackReason).toContain("WebGPU was unavailable");
  });

  it("runs the matrix CPU path", async () => {
    const result = await runExperiment(
      { ...baseConfig, kernel: "matrix-multiply" },
      null,
    );
    expect(result.parameters).toEqual({ size: 16 });
    expect(result.validation.comparedValues).toBe(256);
  });

  it("runs the image convolution CPU path", async () => {
    const result = await runExperiment(
      { ...baseConfig, kernel: "image-convolution", convolutionPreset: "sharpen" },
      null,
    );
    expect(result.parameters.filter).toBe("sharpen");
    expect(result.validation.comparedValues).toBe(32 * 32 * 4);
  });

  it("creates a versioned, local-only export document", async () => {
    const result = await runExperiment(baseConfig, null);
    const document = createExportDocument(result);
    expect(document.schemaVersion).toBe("1.0");
    expect(document.generator).toBe("webgpu-kernel-lab");
    expect(serializeResult(result).endsWith("\n")).toBe(true);
    expect(serializeResult(result)).not.toMatch(/email|contact|location/i);
  });
});
