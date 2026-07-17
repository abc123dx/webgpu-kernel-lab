import {
  type BackendPreference,
  type ExperimentConfig,
  type ExperimentPhase,
  type ExperimentResult,
  runExperiment,
} from "../bench/experiment";
import { downloadResult } from "../bench/export";
import {
  formatMilliseconds,
} from "../bench/statistics";
import {
  imageConvolutionWgsl,
} from "../kernels/image-convolution";
import {
  matrixMultiplyWgsl,
} from "../kernels/matrix-multiply";
import {
  getKernelDescriptor,
  KERNELS,
  type ConvolutionPreset,
  type KernelKind,
} from "../kernels/types";
import { vectorAddWgsl } from "../kernels/vector-add";
import type { WebGpuCapabilityReport } from "../runtime/capabilities";
import {
  initializeWebGpu,
  type WebGpuRuntime,
} from "../runtime/webgpu";

interface AppState {
  config: ExperimentConfig;
  capability: WebGpuCapabilityReport | null;
  result: ExperimentResult | null;
  running: boolean;
  progress: string;
  error: string | null;
}

const DEFAULT_CONFIG: ExperimentConfig = {
  kernel: "vector-add",
  backend: "auto",
  vectorLength: 262_144,
  matrixSize: 64,
  imageSize: 192,
  convolutionPreset: "gaussian-blur",
  warmupIterations: 2,
  benchmarkIterations: 5,
};

export class KernelLabApp {
  readonly #root: HTMLElement;
  #runtime: WebGpuRuntime | null = null;
  #state: AppState = {
    config: { ...DEFAULT_CONFIG },
    capability: null,
    result: null,
    running: false,
    progress: "Initializing device…",
    error: null,
  };

  constructor(root: HTMLElement) {
    this.#root = root;
  }

  async start(): Promise<void> {
    this.#render();
    const initialization = await initializeWebGpu();
    this.#runtime = initialization.runtime;
    this.#state.capability = initialization.report;
    this.#state.progress = "";
    this.#render();
  }

  #render(): void {
    const descriptor = getKernelDescriptor(this.#state.config.kernel);
    this.#root.innerHTML = `
      <div class="ambient ambient-one"></div>
      <div class="ambient ambient-two"></div>
      <header class="topbar">
        <a class="brand" href="#top" aria-label="WebGPU Kernel Lab home">
          <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <span>Kernel Lab</span>
        </a>
        <div class="local-badge">
          <span aria-hidden="true"></span>
          Local execution · zero uploads
        </div>
      </header>

      <main id="top">
        <section class="hero">
          <div class="hero-copy">
            <p class="eyebrow"><span>01</span> Browser compute, made inspectable</p>
            <h1>See the kernel.<br><em>Measure the work.</em></h1>
            <p class="hero-lede">
              Run real WGSL compute beside a typed CPU reference. Warm the pipeline,
              inspect the dispatch, validate every value, then export the evidence.
            </p>
            <div class="hero-actions">
              <a class="button secondary" href="#workbench">Open workbench</a>
              ${renderStatusPill(this.#state.capability)}
            </div>
          </div>
          <div class="hero-orbit" aria-hidden="true">
            <div class="orbit-ring orbit-ring-one"></div>
            <div class="orbit-ring orbit-ring-two"></div>
            <div class="orbit-core">
              <span>WGSL</span>
              <strong>64</strong>
              <small>threads / group</small>
            </div>
            <span class="orbit-label label-a">dispatch</span>
            <span class="orbit-label label-b">validate</span>
            <span class="orbit-label label-c">readback</span>
          </div>
        </section>

        <section class="workbench-section" id="workbench">
          <div class="section-heading">
            <div>
              <p class="eyebrow"><span>02</span> Experiment surface</p>
              <h2>One input. Two paths. No hidden backend.</h2>
            </div>
            <p>The selected CPU and GPU paths receive the same deterministic data.</p>
          </div>

          <div class="kernel-tabs" role="tablist" aria-label="Compute kernel">
            ${KERNELS.map((kernel, index) => `
              <button
                class="kernel-tab ${kernel.kind === this.#state.config.kernel ? "active" : ""}"
                type="button"
                role="tab"
                aria-selected="${kernel.kind === this.#state.config.kernel}"
                data-kernel="${kernel.kind}"
              >
                <span class="tab-index">0${index + 1}</span>
                <strong>${kernel.title}</strong>
                <small>${kernel.shortName}</small>
              </button>
            `).join("")}
          </div>

          <div class="workbench-grid">
            <article class="panel controls-panel">
              <div class="panel-kicker">Configuration</div>
              <div class="kernel-heading">
                <div>
                  <h3>${descriptor.title}</h3>
                  <p>${descriptor.description}</p>
                </div>
                <span class="workgroup-chip">${descriptor.workgroup}</span>
              </div>

              <div class="control-grid">
                <label class="field">
                  <span>Execution path</span>
                  <select id="backend-select" ${this.#state.running ? "disabled" : ""}>
                    ${selectOption("auto", "Auto · prefer WebGPU", this.#state.config.backend)}
                    ${selectOption("webgpu", "WebGPU · fallback if needed", this.#state.config.backend)}
                    ${selectOption("cpu", "CPU reference only", this.#state.config.backend)}
                  </select>
                </label>
                ${this.#renderSizeControl()}
                ${this.#state.config.kernel === "image-convolution" ? `
                  <label class="field">
                    <span>Filter preset</span>
                    <select id="filter-select" ${this.#state.running ? "disabled" : ""}>
                      ${selectOption("gaussian-blur", "Gaussian blur", this.#state.config.convolutionPreset)}
                      ${selectOption("sharpen", "Sharpen", this.#state.config.convolutionPreset)}
                      ${selectOption("edge-detect", "Edge detect", this.#state.config.convolutionPreset)}
                    </select>
                  </label>
                ` : ""}
                <label class="field compact-field">
                  <span>Warmups</span>
                  <input id="warmup-input" type="number" min="0" max="10"
                    value="${this.#state.config.warmupIterations}"
                    ${this.#state.running ? "disabled" : ""}>
                </label>
                <label class="field compact-field">
                  <span>Samples</span>
                  <input id="iterations-input" type="number" min="1" max="30"
                    value="${this.#state.config.benchmarkIterations}"
                    ${this.#state.running ? "disabled" : ""}>
                </label>
              </div>

              <button class="run-button" id="run-button" type="button"
                ${this.#state.running ? "disabled" : ""}>
                <span class="run-icon" aria-hidden="true">${this.#state.running ? "···" : "▶"}</span>
                <span>
                  <strong>${this.#state.running ? "Experiment running" : "Run experiment"}</strong>
                  <small>${this.#state.running ? escapeHtml(this.#state.progress) : "Warm up · benchmark · verify"}</small>
                </span>
              </button>

              ${this.#state.error ? `
                <div class="error-message" role="alert">
                  <strong>Experiment stopped</strong>
                  <span>${escapeHtml(this.#state.error)}</span>
                </div>
              ` : ""}
            </article>

            <article class="panel code-panel">
              <div class="code-toolbar">
                <div class="traffic-lights" aria-hidden="true"><i></i><i></i><i></i></div>
                <span>${this.#state.config.kernel}.wgsl</span>
                <span class="language-label">WGSL</span>
              </div>
              <pre tabindex="0" aria-label="${descriptor.title} WGSL source preview"><code>${escapeHtml(codeFor(this.#state.config.kernel))}</code></pre>
              <div class="code-footer">
                <span><i class="status-dot"></i> Bounds checked</span>
                <span>No shader generation</span>
              </div>
            </article>
          </div>
        </section>

        <section class="results-section" aria-live="polite">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow"><span>03</span> Measurement</p>
              <h2>Evidence, not a victory lap.</h2>
            </div>
            ${this.#state.result ? `
              <button type="button" class="button tertiary" id="export-button">
                Export JSON <span aria-hidden="true">↓</span>
              </button>
            ` : ""}
          </div>
          ${this.#state.result ? renderResult(this.#state.result) : renderEmptyResult(this.#state.running, this.#state.progress)}
        </section>

        <section class="capability-section">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow"><span>04</span> Device boundary</p>
              <h2>What this browser actually exposes.</h2>
            </div>
            <p>Reported locally by the WebGPU adapter and device.</p>
          </div>
          ${renderCapabilities(this.#state.capability)}
        </section>

        <section class="method-section">
          <article>
            <span class="method-number">A</span>
            <h3>Deterministic inputs</h3>
            <p>A seeded generator makes CPU and GPU paths operate on identical values.</p>
          </article>
          <article>
            <span class="method-number">B</span>
            <h3>Honest timing</h3>
            <p>GPU samples include queue submission and mapped-buffer readback, not isolated shader timestamps.</p>
          </article>
          <article>
            <span class="method-number">C</span>
            <h3>Explicit validation</h3>
            <p>Every returned float is compared with the typed CPU reference using a documented tolerance.</p>
          </article>
        </section>
      </main>

      <footer>
        <div class="brand footer-brand">
          <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <span>WebGPU Kernel Lab</span>
        </div>
        <p>Runs entirely in this tab. No account, network API, or telemetry.</p>
        <span>v0.1.0 · MIT</span>
      </footer>
    `;
    this.#bindEvents();
  }

  #renderSizeControl(): string {
    const config = this.#state.config;
    if (config.kernel === "vector-add") {
      return rangeField(
        "size-input",
        "Vector length",
        config.vectorLength,
        65_536,
        1_048_576,
        65_536,
        config.vectorLength.toLocaleString(),
        this.#state.running,
      );
    }
    if (config.kernel === "matrix-multiply") {
      return rangeField(
        "size-input",
        "Square matrix",
        config.matrixSize,
        16,
        128,
        16,
        `${config.matrixSize} × ${config.matrixSize}`,
        this.#state.running,
      );
    }
    return rangeField(
      "size-input",
      "Generated image",
      config.imageSize,
      32,
      512,
      32,
      `${config.imageSize} × ${config.imageSize}`,
      this.#state.running,
    );
  }

  #bindEvents(): void {
    for (const element of this.#root.querySelectorAll<HTMLButtonElement>("[data-kernel]")) {
      element.addEventListener("click", () => {
        if (this.#state.running) return;
        this.#state.config.kernel = element.dataset.kernel as KernelKind;
        this.#state.result = null;
        this.#state.error = null;
        this.#render();
      });
    }

    this.#root.querySelector<HTMLSelectElement>("#backend-select")?.addEventListener(
      "change",
      (event) => {
        this.#state.config.backend = (event.currentTarget as HTMLSelectElement)
          .value as BackendPreference;
      },
    );
    this.#root.querySelector<HTMLSelectElement>("#filter-select")?.addEventListener(
      "change",
      (event) => {
        this.#state.config.convolutionPreset = (event.currentTarget as HTMLSelectElement)
          .value as ConvolutionPreset;
      },
    );
    this.#root.querySelector<HTMLInputElement>("#size-input")?.addEventListener(
      "input",
      (event) => {
        const value = Number((event.currentTarget as HTMLInputElement).value);
        const kind = this.#state.config.kernel;
        if (kind === "vector-add") this.#state.config.vectorLength = value;
        if (kind === "matrix-multiply") this.#state.config.matrixSize = value;
        if (kind === "image-convolution") this.#state.config.imageSize = value;
        const output = this.#root.querySelector<HTMLOutputElement>("#size-output");
        if (output) {
          output.textContent =
            kind === "vector-add"
              ? value.toLocaleString()
              : `${value} × ${value}`;
        }
      },
    );
    this.#root.querySelector<HTMLInputElement>("#warmup-input")?.addEventListener(
      "change",
      (event) => {
        this.#state.config.warmupIterations = Number(
          (event.currentTarget as HTMLInputElement).value,
        );
      },
    );
    this.#root.querySelector<HTMLInputElement>("#iterations-input")?.addEventListener(
      "change",
      (event) => {
        this.#state.config.benchmarkIterations = Number(
          (event.currentTarget as HTMLInputElement).value,
        );
      },
    );
    this.#root.querySelector<HTMLButtonElement>("#run-button")?.addEventListener(
      "click",
      () => {
        void this.#run();
      },
    );
    this.#root.querySelector<HTMLButtonElement>("#export-button")?.addEventListener(
      "click",
      () => {
        if (this.#state.result) downloadResult(this.#state.result);
      },
    );
  }

  async #run(): Promise<void> {
    this.#state.running = true;
    this.#state.error = null;
    this.#state.result = null;
    this.#state.progress = "Preparing deterministic inputs…";
    this.#render();

    try {
      this.#state.result = await runExperiment(
        { ...this.#state.config },
        this.#runtime,
        (phase, completed, total) => this.#updateProgress(phase, completed, total),
      );
    } catch (error) {
      this.#state.error =
        error instanceof Error ? error.message : "An unknown error occurred.";
    } finally {
      this.#state.running = false;
      this.#state.progress = "";
      this.#render();
    }
  }

  #updateProgress(phase: ExperimentPhase, completed: number, total: number): void {
    const labels: Record<ExperimentPhase, string> = {
      preparing: "Preparing deterministic inputs…",
      warming: `Warming pipeline ${completed}/${total}`,
      benchmarking: `Collecting sample ${completed}/${total}`,
      validating: "Comparing every returned value…",
    };
    this.#state.progress = labels[phase];
    const detail = this.#root.querySelector<HTMLElement>(".run-button small");
    if (detail) detail.textContent = this.#state.progress;
    const live = this.#root.querySelector<HTMLElement>(".empty-result p");
    if (live) live.textContent = this.#state.progress;
  }
}

function renderStatusPill(report: WebGpuCapabilityReport | null): string {
  if (!report) {
    return `<span class="status-pill pending"><i></i>Inspecting device</span>`;
  }
  const ready = report.status === "ready";
  return `
    <span class="status-pill ${ready ? "ready" : "fallback"}">
      <i></i>${ready ? "WebGPU ready" : "CPU fallback ready"}
    </span>
  `;
}

function renderEmptyResult(running: boolean, progress: string): string {
  return `
    <div class="empty-result">
      <div class="empty-signal" aria-hidden="true">
        ${Array.from({ length: 18 }, (_, index) => `<i style="--level:${(index * 37) % 91 + 8}%"></i>`).join("")}
      </div>
      <div>
        <strong>${running ? "Measurement in progress" : "No measurement yet"}</strong>
        <p>${escapeHtml(running ? progress : "Choose a kernel and run the experiment. Results stay in this tab until you export them.")}</p>
      </div>
    </div>
  `;
}

function renderResult(result: ExperimentResult): string {
  const speedup =
    result.backend === "webgpu" && result.benchmark.medianMs > 0
      ? result.cpuReference.medianMs / result.benchmark.medianMs
      : null;
  return `
    ${result.fallbackReason ? `<div class="fallback-banner">${escapeHtml(result.fallbackReason)}</div>` : ""}
    <div class="result-grid">
      <article class="metric-card primary-metric">
        <span>Median · ${result.backend.toUpperCase()}</span>
        <strong>${formatMilliseconds(result.benchmark.medianMs)}</strong>
        <small>${result.benchmark.iterations} measured samples</small>
      </article>
      <article class="metric-card">
        <span>CPU reference</span>
        <strong>${formatMilliseconds(result.cpuReference.medianMs)}</strong>
        <small>${speedup === null ? "Selected execution path" : `${formatRatio(speedup)} relative speed`}</small>
      </article>
      <article class="metric-card">
        <span>Validation</span>
        <strong class="${result.validation.passed ? "pass-text" : "fail-text"}">
          ${result.validation.passed ? "PASS" : "FAIL"}
        </strong>
        <small>max error ${formatScientific(result.validation.maximumAbsoluteError)}</small>
      </article>
      <article class="metric-card">
        <span>Throughput</span>
        <strong>${formatThroughput(result.operationsPerSecond)}</strong>
        <small>${result.operationCount.toLocaleString()} modeled operations</small>
      </article>
    </div>
    <div class="result-detail-grid">
      <article class="panel result-detail">
        <div class="panel-kicker">Output fingerprint</div>
        <div class="signal-chart" aria-label="Chart of the first output values">
          ${renderSignalBars(result.outputPreview)}
        </div>
        <div class="preview-values">
          ${result.outputPreview.slice(0, 6).map((value) => `<code>${value}</code>`).join("")}
        </div>
      </article>
      <article class="panel result-detail result-table-wrap">
        <div class="panel-kicker">Run record</div>
        <dl class="run-record">
          <div><dt>Backend</dt><dd>${result.backend.toUpperCase()}</dd></div>
          <div><dt>Min / p95</dt><dd>${formatMilliseconds(result.benchmark.minimumMs)} / ${formatMilliseconds(result.benchmark.p95Ms)}</dd></div>
          <div><dt>Input / output</dt><dd>${formatDataSize(result.inputBytes)} / ${formatDataSize(result.outputBytes)}</dd></div>
          <div><dt>Dispatch</dt><dd>${result.dispatch ? result.dispatch.join(" × ") : "CPU loop"}</dd></div>
          <div><dt>Mismatches</dt><dd>${result.validation.mismatchCount.toLocaleString()} / ${result.validation.comparedValues.toLocaleString()}</dd></div>
        </dl>
      </article>
    </div>
  `;
}

function renderCapabilities(report: WebGpuCapabilityReport | null): string {
  if (!report) {
    return `
      <div class="capability-loading panel">
        <i></i><div><strong>Inspecting the browser device boundary</strong><span>Requesting an adapter without sending data anywhere.</span></div>
      </div>
    `;
  }

  const ready = report.status === "ready";
  return `
    <div class="capability-shell panel">
      <div class="capability-summary ${ready ? "ready" : "fallback"}">
        <div class="capability-icon" aria-hidden="true">${ready ? "◆" : "◇"}</div>
        <div>
          <span>${ready ? "Hardware path" : "Compatibility path"}</span>
          <h3>${escapeHtml(report.headline)}</h3>
          <p>${escapeHtml(report.detail)}</p>
        </div>
      </div>
      ${ready ? `
        <div class="limit-grid">
          ${report.limits.map((limit) => `
            <div>
              <span>${escapeHtml(limit.label)}</span>
              <strong>${escapeHtml(limit.displayValue)}</strong>
            </div>
          `).join("")}
        </div>
        <div class="adapter-row">
          <div>
            <span>Adapter</span>
            <strong>${escapeHtml(adapterLabel(report))}</strong>
          </div>
          <div class="feature-list">
            ${report.features.length > 0
              ? report.features.slice(0, 6).map((feature) => `<span>${escapeHtml(feature)}</span>`).join("")
              : `<span>Core features only</span>`}
          </div>
        </div>
      ` : `
        <div class="fallback-explainer">
          <strong>Nothing is disabled.</strong>
          <p>All three kernels use their CPU reference implementation and preserve validation, warmups, sampling, and JSON export.</p>
        </div>
      `}
    </div>
  `;
}

function rangeField(
  id: string,
  label: string,
  value: number,
  minimum: number,
  maximum: number,
  step: number,
  display: string,
  disabled: boolean,
): string {
  return `
    <label class="field range-field">
      <span>${label}<output id="size-output" for="${id}">${display}</output></span>
      <input id="${id}" type="range" min="${minimum}" max="${maximum}" step="${step}"
        value="${value}" ${disabled ? "disabled" : ""}>
      <small><i>${minimum.toLocaleString()}</i><i>${maximum.toLocaleString()}</i></small>
    </label>
  `;
}

function selectOption(value: string, label: string, selected: string): string {
  return `<option value="${value}" ${value === selected ? "selected" : ""}>${label}</option>`;
}

function codeFor(kind: KernelKind): string {
  const source =
    kind === "vector-add"
      ? vectorAddWgsl
      : kind === "matrix-multiply"
        ? matrixMultiplyWgsl
        : imageConvolutionWgsl;
  const lines = source.trim().split("\n");
  const mainIndex = lines.findIndex((line) => line.includes("@compute"));
  return lines.slice(Math.max(0, mainIndex - 2), mainIndex + 15).join("\n");
}

function renderSignalBars(values: readonly number[]): string {
  const maximum = Math.max(...values.map((value) => Math.abs(value)), 0.000001);
  return values
    .map((value) => {
      const height = 12 + (Math.abs(value) / maximum) * 78;
      const direction = value < 0 ? "negative" : "positive";
      return `<i class="${direction}" style="--height:${height.toFixed(1)}%"></i>`;
    })
    .join("");
}

function adapterLabel(report: WebGpuCapabilityReport): string {
  if (!report.adapter) return "Not reported";
  const parts = [
    report.adapter.description,
    report.adapter.vendor,
    report.adapter.architecture,
  ].filter((value, index, items) => value !== "Not reported" && items.indexOf(value) === index);
  return parts.join(" · ") || "Browser-managed adapter";
}

function formatScientific(value: number): string {
  return value === 0 ? "0" : value.toExponential(2);
}

function formatRatio(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return `${value.toFixed(2)}×`;
}

function formatThroughput(value: number): string {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)} Gop/s`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} Mop/s`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(2)} kop/s`;
  return `${value.toFixed(0)} op/s`;
}

function formatDataSize(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MiB`;
  return `${(bytes / 1024).toFixed(1)} KiB`;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
