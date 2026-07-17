import { clamp01 } from "./data";

export const CONVOLUTION_WORKGROUP_SIZE = 8;

export const imageConvolutionWgsl = /* wgsl */ `
struct Parameters {
  width: u32,
  height: u32,
  _padding0: u32,
  _padding1: u32,
}

@group(0) @binding(0) var<storage, read> inputPixels: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputPixels: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(${CONVOLUTION_WORKGROUP_SIZE}, ${CONVOLUTION_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= parameters.width || id.y >= parameters.height) {
    return;
  }

  var color = vec3<f32>(0.0);
  for (var kernelY = 0u; kernelY < 3u; kernelY += 1u) {
    for (var kernelX = 0u; kernelX < 3u; kernelX += 1u) {
      let sourceX = clamp(i32(id.x) + i32(kernelX) - 1, 0, i32(parameters.width) - 1);
      let sourceY = clamp(i32(id.y) + i32(kernelY) - 1, 0, i32(parameters.height) - 1);
      let sourceOffset = (u32(sourceY) * parameters.width + u32(sourceX)) * 4u;
      let weight = weights[kernelY * 3u + kernelX];
      color += vec3<f32>(
        inputPixels[sourceOffset],
        inputPixels[sourceOffset + 1u],
        inputPixels[sourceOffset + 2u]
      ) * weight;
    }
  }

  let destinationOffset = (id.y * parameters.width + id.x) * 4u;
  outputPixels[destinationOffset] = clamp(color.r, 0.0, 1.0);
  outputPixels[destinationOffset + 1u] = clamp(color.g, 0.0, 1.0);
  outputPixels[destinationOffset + 2u] = clamp(color.b, 0.0, 1.0);
  outputPixels[destinationOffset + 3u] = inputPixels[destinationOffset + 3u];
}
`;

export function imageConvolutionCpu(
  input: Float32Array,
  width: number,
  height: number,
  weights: Float32Array,
): Float32Array {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new RangeError("image dimensions must be positive integers");
  }
  if (input.length !== width * height * 4) {
    throw new RangeError("image input must contain width × height × 4 values");
  }
  if (weights.length !== 9) {
    throw new RangeError("convolution filter must contain 9 weights");
  }

  const output = new Float32Array(input.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const destination = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1) {
        let sum = 0;
        for (let kernelY = 0; kernelY < 3; kernelY += 1) {
          for (let kernelX = 0; kernelX < 3; kernelX += 1) {
            const sourceX = Math.min(width - 1, Math.max(0, x + kernelX - 1));
            const sourceY = Math.min(height - 1, Math.max(0, y + kernelY - 1));
            const source = (sourceY * width + sourceX) * 4 + channel;
            sum += (input[source] ?? 0) * (weights[kernelY * 3 + kernelX] ?? 0);
          }
        }
        output[destination + channel] = clamp01(sum);
      }
      output[destination + 3] = input[destination + 3] ?? 1;
    }
  }
  return output;
}
