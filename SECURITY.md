# Security policy

## Supported versions

Security fixes are applied to the latest release and the current default
branch.

## Reporting a vulnerability

Please use the repository's private GitHub Security Advisory reporting flow.
Include:

- the affected version or commit;
- browser and operating-system family when relevant;
- a minimal reproduction;
- the expected and observed behavior; and
- the potential impact.

Do not include live credentials, personal data, or unrelated device details.
Please do not open a public issue for an unpatched vulnerability.

## Security model

WebGPU Kernel Lab is a static, local browser application:

- it has no server-side component or authentication surface;
- it makes no application network requests;
- it does not execute user-provided WGSL;
- it uses fixed, reviewed shader strings bundled at build time;
- exported JSON excludes adapter identity and full buffers; and
- it destroys per-sample GPU buffers after readback.

Browser and GPU-driver security remain outside the project's control. Keep the
browser and operating system current.
