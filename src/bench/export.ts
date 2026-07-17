import type { ExperimentResult } from "./experiment";

export interface ExportDocument {
  schemaVersion: "1.0";
  generator: "webgpu-kernel-lab";
  note: string;
  result: ExperimentResult;
}

export function createExportDocument(result: ExperimentResult): ExportDocument {
  return {
    schemaVersion: "1.0",
    generator: "webgpu-kernel-lab",
    note:
      "本结果由浏览器本地测量。计时包含命令提交与结果回读，不应在无关设备之间直接比较。",
    result,
  };
}

export function serializeResult(result: ExperimentResult): string {
  return `${JSON.stringify(createExportDocument(result), null, 2)}\n`;
}

export function downloadResult(result: ExperimentResult): void {
  const blob = new Blob([serializeResult(result)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `webgpu-kernel-lab-${result.kernel}-${result.backend}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
