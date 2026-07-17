import {
  CONVOLUTION_WORKGROUP_SIZE,
  imageConvolutionWgsl,
} from "../kernels/image-convolution";
import {
  MATRIX_WORKGROUP_SIZE,
  matrixMultiplyWgsl,
} from "../kernels/matrix-multiply";
import {
  VECTOR_ADD_WORKGROUP_SIZE,
  vectorAddWgsl,
} from "../kernels/vector-add";
import {
  adapterUnavailableReport,
  buildCapabilityReport,
  errorReport,
  type WebGpuCapabilityReport,
  unsupportedReport,
} from "./capabilities";

export interface GpuExecutionResult {
  values: Float32Array;
  elapsedMs: number;
  dispatch: readonly [number, number, number];
}

export interface WebGpuInitialization {
  runtime: WebGpuRuntime | null;
  report: WebGpuCapabilityReport;
}

export async function initializeWebGpu(): Promise<WebGpuInitialization> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    return { runtime: null, report: unsupportedReport() };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      return { runtime: null, report: adapterUnavailableReport() };
    }

    const device = await adapter.requestDevice();
    const runtime = new WebGpuRuntime(device);
    return {
      runtime,
      report: buildCapabilityReport(adapter, device),
    };
  } catch (error) {
    return { runtime: null, report: errorReport(error) };
  }
}

export class WebGpuRuntime {
  readonly #device: GPUDevice;
  readonly #pipelines = new Map<string, Promise<GPUComputePipeline>>();
  #disposed = false;

  constructor(device: GPUDevice) {
    this.#device = device;
  }

  async runVectorAdd(
    inputA: Float32Array,
    inputB: Float32Array,
  ): Promise<GpuExecutionResult> {
    if (inputA.length !== inputB.length || inputA.length === 0) {
      throw new RangeError("vector inputs must have the same non-zero length");
    }
    const dispatch = [
      Math.ceil(inputA.length / VECTOR_ADD_WORKGROUP_SIZE),
      1,
      1,
    ] as const;
    return this.#execute(
      "vector-add",
      vectorAddWgsl,
      [inputA, inputB],
      new Uint32Array([inputA.length, 0, 0, 0]),
      inputA.length,
      dispatch,
    );
  }

  async runMatrixMultiply(
    matrixA: Float32Array,
    matrixB: Float32Array,
    size: number,
  ): Promise<GpuExecutionResult> {
    if (
      !Number.isInteger(size) ||
      size <= 0 ||
      matrixA.length !== size * size ||
      matrixB.length !== size * size
    ) {
      throw new RangeError("matrix inputs must contain size × size elements");
    }
    const groups = Math.ceil(size / MATRIX_WORKGROUP_SIZE);
    return this.#execute(
      "matrix-multiply",
      matrixMultiplyWgsl,
      [matrixA, matrixB],
      new Uint32Array([size, 0, 0, 0]),
      size * size,
      [groups, groups, 1],
    );
  }

  async runImageConvolution(
    input: Float32Array,
    width: number,
    height: number,
    weights: Float32Array,
  ): Promise<GpuExecutionResult> {
    if (
      !Number.isInteger(width) ||
      width <= 0 ||
      !Number.isInteger(height) ||
      height <= 0 ||
      input.length !== width * height * 4 ||
      weights.length !== 9
    ) {
      throw new RangeError("invalid image or convolution filter dimensions");
    }
    return this.#execute(
      "image-convolution",
      imageConvolutionWgsl,
      [input, weights],
      new Uint32Array([width, height, 0, 0]),
      input.length,
      [
        Math.ceil(width / CONVOLUTION_WORKGROUP_SIZE),
        Math.ceil(height / CONVOLUTION_WORKGROUP_SIZE),
        1,
      ],
    );
  }

  dispose(): void {
    if (!this.#disposed) {
      this.#device.destroy();
      this.#pipelines.clear();
      this.#disposed = true;
    }
  }

  async #execute(
    pipelineKey: string,
    shaderCode: string,
    inputs: readonly Float32Array[],
    parameters: Uint32Array,
    outputLength: number,
    dispatch: readonly [number, number, number],
  ): Promise<GpuExecutionResult> {
    if (this.#disposed) {
      throw new Error("WebGPU runtime has been disposed");
    }

    const pipeline = await this.#getPipeline(pipelineKey, shaderCode);
    const inputBuffers = inputs.map((input) =>
      createMappedBuffer(
        this.#device,
        input,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      ),
    );
    const outputBuffer = this.#device.createBuffer({
      size: outputLength * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const parameterBuffer = createMappedBuffer(
      this.#device,
      parameters,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    const readbackBuffer = this.#device.createBuffer({
      size: outputLength * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const entries: GPUBindGroupEntry[] = inputBuffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    }));
    entries.push(
      { binding: inputBuffers.length, resource: { buffer: outputBuffer } },
      { binding: inputBuffers.length + 1, resource: { buffer: parameterBuffer } },
    );

    const bindGroup = this.#device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    const encoder = this.#device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatch);
    pass.end();
    encoder.copyBufferToBuffer(
      outputBuffer,
      0,
      readbackBuffer,
      0,
      outputLength * Float32Array.BYTES_PER_ELEMENT,
    );

    const startedAt = performance.now();
    this.#device.queue.submit([encoder.finish()]);
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const elapsedMs = performance.now() - startedAt;
    const values = new Float32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();

    for (const buffer of inputBuffers) {
      buffer.destroy();
    }
    outputBuffer.destroy();
    parameterBuffer.destroy();
    readbackBuffer.destroy();

    return { values, elapsedMs, dispatch };
  }

  #getPipeline(key: string, code: string): Promise<GPUComputePipeline> {
    const cached = this.#pipelines.get(key);
    if (cached) {
      return cached;
    }
    const shader = this.#device.createShaderModule({ code });
    const pipeline = this.#device.createComputePipelineAsync({
      layout: "auto",
      compute: {
        module: shader,
        entryPoint: "main",
      },
    });
    this.#pipelines.set(key, pipeline);
    return pipeline;
  }
}

function createMappedBuffer(
  device: GPUDevice,
  data: Float32Array | Uint32Array,
  usage: GPUBufferUsageFlags,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage,
    mappedAtCreation: true,
  });
  const target = new Uint8Array(buffer.getMappedRange());
  target.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
}
