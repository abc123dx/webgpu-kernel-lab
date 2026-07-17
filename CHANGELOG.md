# 更新日志

所有重要变更都会记录在此。

## [0.1.1] - 2026-07-17

### 中文本地化

- 将实验台界面、状态、指标说明、错误提示和设备能力报告改为中文优先。
- 将 README、演示指南和测量方法完整汉化。
- 保持内核 ID、JSON 字段、WGSL 源码和公开 API 兼容。
- JSON 导出说明改为中文，不改变 `1.0` 文档结构。

## [0.1.0] - 2026-07-17

### Added

- Responsive local WebGPU compute workbench.
- Device capability and limit inspection.
- WGSL vector addition, matrix multiplication, and image convolution kernels.
- Typed CPU references and automatic fallback.
- Adjustable input sizes, warmups, and measured samples.
- End-to-end timing summaries and modeled throughput.
- Value-by-value tolerance validation.
- Versioned JSON export without adapter identity.
- Thirty-nine unit and orchestration tests.
- Compatibility, methodology, security, and contribution documentation.
