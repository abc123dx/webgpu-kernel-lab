export type KernelKind =
  | "vector-add"
  | "matrix-multiply"
  | "image-convolution";

export type ConvolutionPreset = "gaussian-blur" | "sharpen" | "edge-detect";

export interface KernelDescriptor {
  kind: KernelKind;
  shortName: string;
  title: string;
  description: string;
  workgroup: string;
  accent: string;
}

export const KERNELS: readonly KernelDescriptor[] = [
  {
    kind: "vector-add",
    shortName: "AX + BY",
    title: "向量加法",
    description: "带边界保护的一维并行加法。",
    workgroup: "64 × 1 × 1",
    accent: "lime",
  },
  {
    kind: "matrix-multiply",
    shortName: "A × B",
    title: "矩阵乘法",
    description: "在二维网格上执行方阵乘法。",
    workgroup: "8 × 8 × 1",
    accent: "cyan",
  },
  {
    kind: "image-convolution",
    shortName: "3 × 3",
    title: "图像卷积",
    description: "对生成的 RGBA 图像应用带边缘钳制的 3 × 3 滤镜。",
    workgroup: "8 × 8 × 1",
    accent: "violet",
  },
] as const;

export function getKernelDescriptor(kind: KernelKind): KernelDescriptor {
  const descriptor = KERNELS.find((candidate) => candidate.kind === kind);
  if (!descriptor) {
    throw new Error(`未知内核：${kind}`);
  }
  return descriptor;
}
