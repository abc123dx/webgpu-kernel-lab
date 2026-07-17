import { describe, expect, it } from "vitest";
import {
  CONVOLUTION_FILTERS,
  clamp01,
  createMatrix,
  createSequence,
  createTestImage,
} from "../src/kernels/data";

describe("deterministic data generation", () => {
  it("repeats a sequence for the same seed", () => {
    expect(createSequence(8, 42)).toEqual(createSequence(8, 42));
  });

  it("changes the sequence when the seed changes", () => {
    expect(createSequence(8, 42)).not.toEqual(createSequence(8, 43));
  });

  it("keeps generated sequence values in [-1, 1]", () => {
    expect(Array.from(createSequence(1_000)).every((value) => value >= -1 && value <= 1))
      .toBe(true);
  });

  it("rejects invalid sequence lengths", () => {
    expect(() => createSequence(-1)).toThrow(RangeError);
    expect(() => createSequence(1.5)).toThrow(RangeError);
  });

  it("creates square matrices", () => {
    expect(createMatrix(7, 1)).toHaveLength(49);
  });

  it("creates opaque RGBA test images", () => {
    const image = createTestImage(4);
    expect(image).toHaveLength(64);
    for (let offset = 3; offset < image.length; offset += 4) {
      expect(image[offset]).toBe(1);
    }
  });

  it("ships normalized Gaussian weights", () => {
    const sum = CONVOLUTION_FILTERS["gaussian-blur"].reduce(
      (total, value) => total + value,
      0,
    );
    expect(sum).toBeCloseTo(1, 7);
  });

  it("clamps values to the unit interval", () => {
    expect(clamp01(-4)).toBe(0);
    expect(clamp01(0.4)).toBe(0.4);
    expect(clamp01(8)).toBe(1);
  });
});
