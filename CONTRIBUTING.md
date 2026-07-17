# Contributing

Contributions that improve correctness, accessibility, browser compatibility,
tests, or explanations are welcome.

## Development

```bash
npm install
npm run dev
```

Before proposing a change:

```bash
npm run check
```

## Expectations

- Keep computation local and deterministic.
- Add or update a CPU reference before changing a WGSL kernel.
- Add tests for numerical behavior, invalid inputs, and new export fields.
- Preserve the friendly CPU fallback.
- Do not add analytics, remote assets, secret-bearing examples, or personal
  information.
- Document timing boundaries and avoid unsupported performance claims.
- Keep exported schemas backward compatible or explicitly version them.

Small, focused changes are easier to review. Explain the behavior being changed
and include reproducible validation steps.
