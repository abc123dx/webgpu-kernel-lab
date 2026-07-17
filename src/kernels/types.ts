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
    title: "Vector add",
    description: "One-dimensional parallel addition with bounds-safe dispatch.",
    workgroup: "64 × 1 × 1",
    accent: "lime",
  },
  {
    kind: "matrix-multiply",
    shortName: "A × B",
    title: "Matrix multiply",
    description: "Square matrix multiplication across a two-dimensional grid.",
    workgroup: "8 × 8 × 1",
    accent: "cyan",
  },
  {
    kind: "image-convolution",
    shortName: "3 × 3",
    title: "Image convolution",
    description: "A clamped 3 × 3 filter over a generated RGBA image.",
    workgroup: "8 × 8 × 1",
    accent: "violet",
  },
] as const;

export function getKernelDescriptor(kind: KernelKind): KernelDescriptor {
  const descriptor = KERNELS.find((candidate) => candidate.kind === kind);
  if (!descriptor) {
    throw new Error(`Unknown kernel: ${kind}`);
  }
  return descriptor;
}
