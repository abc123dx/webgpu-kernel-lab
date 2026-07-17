export const MATRIX_WORKGROUP_SIZE = 8;

export const matrixMultiplyWgsl = /* wgsl */ `
struct Parameters {
  size: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> matrixA: array<f32>;
@group(0) @binding(1) var<storage, read> matrixB: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(${MATRIX_WORKGROUP_SIZE}, ${MATRIX_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let column = id.x;
  let row = id.y;
  let size = parameters.size;
  if (column >= size || row >= size) {
    return;
  }

  var sum = 0.0;
  for (var inner = 0u; inner < size; inner += 1u) {
    sum += matrixA[row * size + inner] * matrixB[inner * size + column];
  }
  output[row * size + column] = sum;
}
`;

export function matrixMultiplyCpu(
  matrixA: Float32Array,
  matrixB: Float32Array,
  size: number,
): Float32Array {
  if (!Number.isInteger(size) || size <= 0) {
    throw new RangeError("矩阵规模必须是正整数");
  }
  if (matrixA.length !== size * size || matrixB.length !== size * size) {
    throw new RangeError("矩阵输入必须包含 size × size 个元素");
  }

  const output = new Float32Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      let sum = 0;
      for (let inner = 0; inner < size; inner += 1) {
        sum +=
          (matrixA[row * size + inner] ?? 0) *
          (matrixB[inner * size + column] ?? 0);
      }
      output[row * size + column] = sum;
    }
  }
  return output;
}
