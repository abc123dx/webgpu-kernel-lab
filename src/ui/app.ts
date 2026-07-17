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
    progress: "正在初始化设备…",
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
        <a class="brand" href="#top" aria-label="返回 WebGPU 内核实验室首页">
          <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <span>内核实验室</span>
        </a>
        <div class="local-badge">
          <span aria-hidden="true"></span>
          本地运行 · 零上传
        </div>
      </header>

      <main id="top">
        <section class="hero">
          <div class="hero-copy">
            <p class="eyebrow"><span>01</span> 看得见的浏览器计算</p>
            <h1>读懂内核。<br><em>量化计算。</em></h1>
            <p class="hero-lede">
              让真实 WGSL 计算与类型明确的 CPU 参考实现并行运行。预热流水线、
              检查调度、逐值校验，最后导出完整证据。
            </p>
            <div class="hero-actions">
              <a class="button secondary" href="#workbench">打开实验台</a>
              ${renderStatusPill(this.#state.capability)}
            </div>
          </div>
          <div class="hero-orbit" aria-hidden="true">
            <div class="orbit-ring orbit-ring-one"></div>
            <div class="orbit-ring orbit-ring-two"></div>
            <div class="orbit-core">
              <span>WGSL</span>
              <strong>64</strong>
              <small>线程 / 工作组</small>
            </div>
            <span class="orbit-label label-a">调度</span>
            <span class="orbit-label label-b">校验</span>
            <span class="orbit-label label-c">回读</span>
          </div>
        </section>

        <section class="workbench-section" id="workbench">
          <div class="section-heading">
            <div>
              <p class="eyebrow"><span>02</span> 实验工作区</p>
              <h2>同一份输入，两条执行路径，没有隐藏后端。</h2>
            </div>
            <p>所选 CPU 与 GPU 路径接收完全相同的确定性数据。</p>
          </div>

          <div class="kernel-tabs" role="tablist" aria-label="计算内核">
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
              <div class="panel-kicker">实验配置</div>
              <div class="kernel-heading">
                <div>
                  <h3>${descriptor.title}</h3>
                  <p>${descriptor.description}</p>
                </div>
                <span class="workgroup-chip">${descriptor.workgroup}</span>
              </div>

              <div class="control-grid">
                <label class="field">
                  <span>执行路径</span>
                  <select id="backend-select" ${this.#state.running ? "disabled" : ""}>
                    ${selectOption("auto", "自动 · 优先 WebGPU", this.#state.config.backend)}
                    ${selectOption("webgpu", "WebGPU · 不可用时回退", this.#state.config.backend)}
                    ${selectOption("cpu", "仅使用 CPU 参考实现", this.#state.config.backend)}
                  </select>
                </label>
                ${this.#renderSizeControl()}
                ${this.#state.config.kernel === "image-convolution" ? `
                  <label class="field">
                    <span>滤镜预设</span>
                    <select id="filter-select" ${this.#state.running ? "disabled" : ""}>
                      ${selectOption("gaussian-blur", "高斯模糊", this.#state.config.convolutionPreset)}
                      ${selectOption("sharpen", "锐化", this.#state.config.convolutionPreset)}
                      ${selectOption("edge-detect", "边缘检测", this.#state.config.convolutionPreset)}
                    </select>
                  </label>
                ` : ""}
                <label class="field compact-field">
                  <span>预热次数</span>
                  <input id="warmup-input" type="number" min="0" max="10"
                    value="${this.#state.config.warmupIterations}"
                    ${this.#state.running ? "disabled" : ""}>
                </label>
                <label class="field compact-field">
                  <span>测量次数</span>
                  <input id="iterations-input" type="number" min="1" max="30"
                    value="${this.#state.config.benchmarkIterations}"
                    ${this.#state.running ? "disabled" : ""}>
                </label>
              </div>

              <button class="run-button" id="run-button" type="button"
                ${this.#state.running ? "disabled" : ""}>
                <span class="run-icon" aria-hidden="true">${this.#state.running ? "···" : "▶"}</span>
                <span>
                  <strong>${this.#state.running ? "实验运行中" : "运行实验"}</strong>
                  <small>${this.#state.running ? escapeHtml(this.#state.progress) : "预热 · 测量 · 校验"}</small>
                </span>
              </button>

              ${this.#state.error ? `
                <div class="error-message" role="alert">
                  <strong>实验已停止</strong>
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
              <pre tabindex="0" aria-label="${descriptor.title} WGSL 源码预览"><code>${escapeHtml(codeFor(this.#state.config.kernel))}</code></pre>
              <div class="code-footer">
                <span><i class="status-dot"></i> 已进行边界检查</span>
                <span>不动态生成着色器</span>
              </div>
            </article>
          </div>
        </section>

        <section class="results-section" aria-live="polite">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow"><span>03</span> 测量结果</p>
              <h2>用证据说话，不做性能炫技。</h2>
            </div>
            ${this.#state.result ? `
              <button type="button" class="button tertiary" id="export-button">
                导出 JSON <span aria-hidden="true">↓</span>
              </button>
            ` : ""}
          </div>
          ${this.#state.result ? renderResult(this.#state.result) : renderEmptyResult(this.#state.running, this.#state.progress)}
        </section>

        <section class="capability-section">
          <div class="section-heading compact">
            <div>
              <p class="eyebrow"><span>04</span> 设备能力边界</p>
              <h2>查看当前浏览器真正开放了哪些能力。</h2>
            </div>
            <p>数据由本地 WebGPU 适配器与设备直接报告。</p>
          </div>
          ${renderCapabilities(this.#state.capability)}
        </section>

        <section class="method-section">
          <article>
            <span class="method-number">A</span>
            <h3>确定性输入</h3>
            <p>固定种子的生成器确保 CPU 与 GPU 路径处理完全相同的数值。</p>
          </article>
          <article>
            <span class="method-number">B</span>
            <h3>如实计时</h3>
            <p>GPU 测量包含队列提交和映射缓冲区回读，而不是孤立的着色器时间。</p>
          </article>
          <article>
            <span class="method-number">C</span>
            <h3>明确校验</h3>
            <p>按照文档中的误差容限，将每个返回浮点数与 CPU 参考结果逐一比较。</p>
          </article>
        </section>
      </main>

      <footer>
        <div class="brand footer-brand">
          <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i></span>
          <span>WebGPU 内核实验室</span>
        </div>
        <p>完全在当前标签页运行，无需账号，不调用网络 API，也不收集遥测数据。</p>
        <span>v0.1.1 · MIT</span>
      </footer>
    `;
    this.#bindEvents();
  }

  #renderSizeControl(): string {
    const config = this.#state.config;
    if (config.kernel === "vector-add") {
      return rangeField(
        "size-input",
        "向量长度",
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
        "方阵规模",
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
      "生成图像尺寸",
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
    this.#state.progress = "正在准备确定性输入…";
    this.#render();

    try {
      this.#state.result = await runExperiment(
        { ...this.#state.config },
        this.#runtime,
        (phase, completed, total) => this.#updateProgress(phase, completed, total),
      );
    } catch (error) {
      this.#state.error =
        error instanceof Error ? error.message : "发生未知错误。";
    } finally {
      this.#state.running = false;
      this.#state.progress = "";
      this.#render();
    }
  }

  #updateProgress(phase: ExperimentPhase, completed: number, total: number): void {
    const labels: Record<ExperimentPhase, string> = {
      preparing: "正在准备确定性输入…",
      warming: `正在预热流水线 ${completed}/${total}`,
      benchmarking: `正在采集测量样本 ${completed}/${total}`,
      validating: "正在逐一比较返回值…",
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
    return `<span class="status-pill pending"><i></i>正在检查设备</span>`;
  }
  const ready = report.status === "ready";
  return `
    <span class="status-pill ${ready ? "ready" : "fallback"}">
      <i></i>${ready ? "WebGPU 已就绪" : "CPU 回退已就绪"}
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
        <strong>${running ? "正在测量" : "尚无测量结果"}</strong>
        <p>${escapeHtml(running ? progress : "选择一个内核并运行实验。结果只保留在当前标签页，除非你主动导出。")}</p>
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
        <span>中位数 · ${result.backend.toUpperCase()}</span>
        <strong>${formatMilliseconds(result.benchmark.medianMs)}</strong>
        <small>${result.benchmark.iterations} 个有效测量样本</small>
      </article>
      <article class="metric-card">
        <span>CPU 参考结果</span>
        <strong>${formatMilliseconds(result.cpuReference.medianMs)}</strong>
        <small>${speedup === null ? "当前所选执行路径" : `相对速度 ${formatRatio(speedup)}`}</small>
      </article>
      <article class="metric-card">
        <span>数值校验</span>
        <strong class="${result.validation.passed ? "pass-text" : "fail-text"}">
          ${result.validation.passed ? "通过" : "未通过"}
        </strong>
        <small>最大误差 ${formatScientific(result.validation.maximumAbsoluteError)}</small>
      </article>
      <article class="metric-card">
        <span>吞吐量</span>
        <strong>${formatThroughput(result.operationsPerSecond)}</strong>
        <small>${result.operationCount.toLocaleString()} 次模型化运算</small>
      </article>
    </div>
    <div class="result-detail-grid">
      <article class="panel result-detail">
        <div class="panel-kicker">输出指纹</div>
        <div class="signal-chart" aria-label="前若干个输出值的柱状图">
          ${renderSignalBars(result.outputPreview)}
        </div>
        <div class="preview-values">
          ${result.outputPreview.slice(0, 6).map((value) => `<code>${value}</code>`).join("")}
        </div>
      </article>
      <article class="panel result-detail result-table-wrap">
        <div class="panel-kicker">运行记录</div>
        <dl class="run-record">
          <div><dt>执行后端</dt><dd>${result.backend.toUpperCase()}</dd></div>
          <div><dt>最小值 / p95</dt><dd>${formatMilliseconds(result.benchmark.minimumMs)} / ${formatMilliseconds(result.benchmark.p95Ms)}</dd></div>
          <div><dt>输入 / 输出</dt><dd>${formatDataSize(result.inputBytes)} / ${formatDataSize(result.outputBytes)}</dd></div>
          <div><dt>调度规模</dt><dd>${result.dispatch ? result.dispatch.join(" × ") : "CPU 循环"}</dd></div>
          <div><dt>不匹配数量</dt><dd>${result.validation.mismatchCount.toLocaleString()} / ${result.validation.comparedValues.toLocaleString()}</dd></div>
        </dl>
      </article>
    </div>
  `;
}

function renderCapabilities(report: WebGpuCapabilityReport | null): string {
  if (!report) {
    return `
      <div class="capability-loading panel">
        <i></i><div><strong>正在检查浏览器设备能力</strong><span>只在本地请求适配器，不向任何位置发送数据。</span></div>
      </div>
    `;
  }

  const ready = report.status === "ready";
  return `
    <div class="capability-shell panel">
      <div class="capability-summary ${ready ? "ready" : "fallback"}">
        <div class="capability-icon" aria-hidden="true">${ready ? "◆" : "◇"}</div>
        <div>
          <span>${ready ? "硬件执行路径" : "兼容执行路径"}</span>
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
            <span>适配器</span>
            <strong>${escapeHtml(adapterLabel(report))}</strong>
          </div>
          <div class="feature-list">
            ${report.features.length > 0
              ? report.features.slice(0, 6).map((feature) => `<span>${escapeHtml(feature)}</span>`).join("")
              : `<span>仅核心功能</span>`}
          </div>
        </div>
      ` : `
        <div class="fallback-explainer">
          <strong>所有实验仍可运行。</strong>
          <p>三个内核都会改用 CPU 参考实现，并继续提供数值校验、预热、重复测量和 JSON 导出。</p>
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
  if (!report.adapter) return "未报告";
  const parts = [
    report.adapter.description,
    report.adapter.vendor,
    report.adapter.architecture,
  ].filter((value, index, items) => value !== "未报告" && items.indexOf(value) === index);
  return parts.join(" · ") || "浏览器管理的适配器";
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
