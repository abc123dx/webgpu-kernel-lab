export const VECTOR_ADD_WORKGROUP_SIZE = 64;

export const vectorAddWgsl = /* wgsl */ `
struct Parameters {
  length: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}

@group(0) @binding(0) var<storage, read> inputA: array<f32>;
@group(0) @binding(1) var<storage, read> inputB: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

@compute @workgroup_size(${VECTOR_ADD_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= parameters.length) {
    return;
  }
  output[index] = inputA[index] + inputB[index];
}
`;

export function vectorAddCpu(
  inputA: Float32Array,
  inputB: Float32Array,
): Float32Array {
  if (inputA.length !== inputB.length) {
    throw new RangeError("两个向量输入必须具有相同长度");
  }

  const output = new Float32Array(inputA.length);
  for (let index = 0; index < inputA.length; index += 1) {
    output[index] = (inputA[index] ?? 0) + (inputB[index] ?? 0);
  }
  return output;
}
