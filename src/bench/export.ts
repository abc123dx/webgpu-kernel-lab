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
      "Local browser measurement. Timings include submission and readback and are not comparable across unrelated devices.",
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
