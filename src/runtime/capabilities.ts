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
    headline: "CPU 回退模式已启用",
    detail:
      "当前浏览器未开放 WebGPU。所有实验仍可通过类型明确的 CPU 参考路径运行。",
    adapter: null,
    features: [],
    limits: [],
  };
}

export function adapterUnavailableReport(): WebGpuCapabilityReport {
  return {
    status: "adapter-unavailable",
    headline: "未找到兼容的 GPU 适配器",
    detail:
      "浏览器支持 WebGPU，但无法提供适配器。CPU 回退路径已就绪，并会使用相同的生成输入。",
    adapter: null,
    features: [],
    limits: [],
  };
}

export function errorReport(error: unknown): WebGpuCapabilityReport {
  return {
    status: "error",
    headline: "WebGPU 初始化失败",
    detail: error instanceof Error ? error.message : "初始化过程中发生未知错误。",
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
    ["最大存储缓冲区", device.limits.maxStorageBufferBindingSize],
    ["最大缓冲区大小", device.limits.maxBufferSize],
    ["工作组最大调用数", device.limits.maxComputeInvocationsPerWorkgroup],
    ["工作组存储空间", device.limits.maxComputeWorkgroupStorageSize],
  ] as const;

  return {
    status: "ready",
    headline: "WebGPU 计算已就绪",
    detail:
      "设备已在本地完成初始化。测量时间包含命令提交和结果回读，而不只是着色器执行时间。",
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
      displayValue: label.includes("缓冲区") || label.includes("存储")
        ? formatBytes(value)
        : value.toLocaleString(),
    })),
  };
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    throw new RangeError("字节数必须是有限的非负数");
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
  return value.trim() || "未报告";
}
