import type { ConvolutionPreset } from "./types";

export const CONVOLUTION_FILTERS: Record<ConvolutionPreset, Float32Array> = {
  "gaussian-blur": new Float32Array([
    1 / 16, 2 / 16, 1 / 16,
    2 / 16, 4 / 16, 2 / 16,
    1 / 16, 2 / 16, 1 / 16,
  ]),
  sharpen: new Float32Array([
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0,
  ]),
  "edge-detect": new Float32Array([
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1,
  ]),
};

export function createSequence(length: number, seed = 0x1234abcd): Float32Array {
  if (!Number.isInteger(length) || length < 0) {
    throw new RangeError("长度必须是非负整数");
  }

  let state = seed >>> 0;
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    output[index] = (state / 0xffffffff) * 2 - 1;
  }
  return output;
}

export function createMatrix(size: number, seed: number): Float32Array {
  assertPositiveInteger(size, "矩阵规模");
  return createSequence(size * size, seed);
}

export function createTestImage(size: number): Float32Array {
  assertPositiveInteger(size, "图像尺寸");
  const pixels = new Float32Array(size * size * 4);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * 4;
      const nx = x / Math.max(1, size - 1);
      const ny = y / Math.max(1, size - 1);
      const checker = (Math.floor(x / 12) + Math.floor(y / 12)) % 2;
      pixels[offset] = clamp01(0.12 + nx * 0.72 + checker * 0.12);
      pixels[offset + 1] = clamp01(0.08 + ny * 0.76);
      pixels[offset + 2] = clamp01(0.16 + (1 - nx) * 0.46 + checker * 0.1);
      pixels[offset + 3] = 1;
    }
  }

  return pixels;
}

export function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label}必须是正整数`);
  }
}

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}
