import { describe, expect, it } from "vitest";
import { imageConvolutionCpu, imageConvolutionWgsl } from "../src/kernels/image-convolution";
import { matrixMultiplyCpu, matrixMultiplyWgsl } from "../src/kernels/matrix-multiply";
import { vectorAddCpu, vectorAddWgsl } from "../src/kernels/vector-add";

describe("CPU reference kernels", () => {
  it("adds two vectors", () => {
    expect(vectorAddCpu(
      new Float32Array([1, -2, 3.5]),
      new Float32Array([4, 2, 0.5]),
    )).toEqual(new Float32Array([5, 0, 4]));
  });

  it("rejects mismatched vectors", () => {
    expect(() =>
      vectorAddCpu(new Float32Array([1]), new Float32Array([1, 2])),
    ).toThrow(RangeError);
  });

  it("multiplies a known 2 × 2 matrix", () => {
    const result = matrixMultiplyCpu(
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      2,
    );
    expect(result).toEqual(new Float32Array([19, 22, 43, 50]));
  });

  it("preserves a matrix multiplied by identity", () => {
    const matrix = new Float32Array([2, 4, 6, 8]);
    const identity = new Float32Array([1, 0, 0, 1]);
    expect(matrixMultiplyCpu(matrix, identity, 2)).toEqual(matrix);
  });

  it("rejects malformed matrices", () => {
    expect(() =>
      matrixMultiplyCpu(new Float32Array([1]), new Float32Array([1]), 2),
    ).toThrow(RangeError);
  });

  it("preserves an image through an identity convolution", () => {
    const image = new Float32Array([
      0.2, 0.3, 0.4, 1,
      0.5, 0.6, 0.7, 1,
      0.8, 0.9, 1, 1,
      0.1, 0.2, 0.3, 1,
    ]);
    const identity = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
    expect(imageConvolutionCpu(image, 2, 2, identity)).toEqual(image);
  });

  it("clamps high convolution output", () => {
    const image = new Float32Array([0.8, 0.7, 0.6, 1]);
    const boost = new Float32Array([0, 0, 0, 0, 2, 0, 0, 0, 0]);
    expect(imageConvolutionCpu(image, 1, 1, boost)).toEqual(
      new Float32Array([1, 1, 1, 1]),
    );
  });

  it("rejects malformed convolution filters", () => {
    expect(() =>
      imageConvolutionCpu(
        new Float32Array([0, 0, 0, 1]),
        1,
        1,
        new Float32Array([1]),
      ),
    ).toThrow(RangeError);
  });
});

describe("WGSL kernels", () => {
  it.each([
    ["vector", vectorAddWgsl],
    ["matrix", matrixMultiplyWgsl],
    ["convolution", imageConvolutionWgsl],
  ])("declares a bounds-safe %s compute entry point", (_name, source) => {
    expect(source).toContain("@compute");
    expect(source).toContain("workgroup_size");
    expect(source).toMatch(/if \(.+>=/);
  });
});
