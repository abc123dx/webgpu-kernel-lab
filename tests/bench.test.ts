import { describe, expect, it } from "vitest";
import {
  calculateThroughput,
  formatMilliseconds,
  percentile,
  summarizeDurations,
} from "../src/bench/statistics";
import { compareFloatArrays } from "../src/bench/validation";
import { formatBytes } from "../src/runtime/capabilities";

describe("benchmark statistics", () => {
  it("summarizes sorted and unsorted durations", () => {
    const result = summarizeDurations([9, 1, 5, 3]);
    expect(result.minimumMs).toBe(1);
    expect(result.maximumMs).toBe(9);
    expect(result.meanMs).toBe(4.5);
    expect(result.medianMs).toBe(4);
  });

  it("interpolates percentiles", () => {
    expect(percentile([0, 10], 0.25)).toBe(2.5);
  });

  it("rejects empty samples", () => {
    expect(() => summarizeDurations([])).toThrow(RangeError);
  });

  it("rejects negative duration samples", () => {
    expect(() => summarizeDurations([1, -1])).toThrow(RangeError);
  });

  it("calculates operations per second", () => {
    expect(calculateThroughput(2_000, 2)).toBe(1_000_000);
  });

  it("formats very short measurements honestly", () => {
    expect(formatMilliseconds(0.001)).toBe("<0.01 ms");
  });
});

describe("numeric validation", () => {
  it("passes identical arrays", () => {
    const result = compareFloatArrays(
      new Float32Array([1, 2]),
      new Float32Array([1, 2]),
    );
    expect(result.passed).toBe(true);
    expect(result.maximumAbsoluteError).toBe(0);
  });

  it("counts values outside tolerance", () => {
    const result = compareFloatArrays(
      new Float32Array([1, 2.2, 3]),
      new Float32Array([1, 2, 3.1]),
      0.05,
    );
    expect(result.passed).toBe(false);
    expect(result.mismatchCount).toBe(2);
    expect(result.meanAbsoluteError).toBeGreaterThan(0);
  });

  it("accepts values exactly at tolerance", () => {
    const result = compareFloatArrays(
      new Float32Array([1.1]),
      new Float32Array([1]),
      0.100001,
    );
    expect(result.passed).toBe(true);
  });

  it("rejects arrays with different lengths", () => {
    expect(() =>
      compareFloatArrays(new Float32Array([1]), new Float32Array([1, 2])),
    ).toThrow(RangeError);
  });

  it("rejects negative tolerance", () => {
    expect(() =>
      compareFloatArrays(new Float32Array(), new Float32Array(), -1),
    ).toThrow(RangeError);
  });
});

describe("capability formatting", () => {
  it("formats binary data sizes", () => {
    expect(formatBytes(1_048_576)).toBe("1.0 MiB");
    expect(formatBytes(512)).toBe("512 B");
  });
});
