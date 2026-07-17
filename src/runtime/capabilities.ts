export type WebGpuStatus = "ready" | "unsupported" | "adapter-unavailable" | "error";

export interface CapabilityLimit {
  label: string;
  value: number;
  displayValue: string;
}

export interface WebGpuCapabilityReport {
  status: WebGpuStatus;
  headline: string;
  detail: string;
  adapter: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
  } | null;
  features: string[];
  limits: CapabilityLimit[];
}

export function unsupportedReport(): WebGpuCapabilityReport {
  return {
    status: "unsupported",
    headline: "CPU fallback active",
    detail:
      "This browser does not expose WebGPU. Every experiment remains available through the typed CPU reference path.",
    adapter: null,
    features: [],
    limits: [],
  };
}

export function adapterUnavailableReport(): WebGpuCapabilityReport {
  return {
    status: "adapter-unavailable",
    headline: "No compatible GPU adapter",
    detail:
      "WebGPU exists, but the browser could not provide an adapter. CPU fallback is ready and uses the same generated inputs.",
    adapter: null,
    features: [],
    limits: [],
  };
}

export function errorReport(error: unknown): WebGpuCapabilityReport {
  return {
    status: "error",
    headline: "WebGPU initialization failed",
    detail: error instanceof Error ? error.message : "An unknown initialization error occurred.",
    adapter: null,
    features: [],
    limits: [],
  };
}

export function buildCapabilityReport(
  adapter: GPUAdapter,
  device: GPUDevice,
): WebGpuCapabilityReport {
  const info = adapter.info;
  const rawLimits = [
    ["Max storage buffer", device.limits.maxStorageBufferBindingSize],
    ["Max buffer size", device.limits.maxBufferSize],
    ["Workgroup invocations", device.limits.maxComputeInvocationsPerWorkgroup],
    ["Workgroup storage", device.limits.maxComputeWorkgroupStorageSize],
  ] as const;

  return {
    status: "ready",
    headline: "WebGPU compute ready",
    detail:
      "The device is initialized locally. Benchmarks include command submission and result readback, not only shader time.",
    adapter: {
      vendor: normalizeAdapterInfo(info.vendor),
      architecture: normalizeAdapterInfo(info.architecture),
      device: normalizeAdapterInfo(info.device),
      description: normalizeAdapterInfo(info.description),
    },
    features: [...device.features].sort(),
    limits: rawLimits.map(([label, value]) => ({
      label,
      value,
      displayValue: label.includes("buffer") || label.includes("storage")
        ? formatBytes(value)
        : value.toLocaleString(),
    })),
  };
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    throw new RangeError("bytes must be a finite non-negative number");
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = units[0] ?? "KiB";
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index] ?? unit;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${unit}`;
}

function normalizeAdapterInfo(value: string): string {
  return value.trim() || "Not reported";
}
