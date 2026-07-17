# Demo walkthrough

This walkthrough demonstrates every user-visible path without an account or
external service.

## 1. Inspect the device boundary

Open the local app and wait for the status pill to settle:

- **WebGPU ready** means the browser returned an adapter and device.
- **CPU fallback ready** means the app could not initialize WebGPU. This is a
  supported mode, not an error page.

The device section shows only values supplied by the local WebGPU API. Optional
features may be empty on a fully functional core device.

## 2. Run vector addition

1. Keep **Vector add** selected.
2. Choose **Auto · prefer WebGPU**.
3. Set the vector to `262,144`, warmups to `2`, and samples to `5`.
4. Select **Run experiment**.

The result should report:

- the actual CPU or WebGPU backend;
- a passing comparison;
- zero or near-zero maximum absolute error;
- a `64`-wide workgroup dispatch when WebGPU ran; and
- the first twelve output values as a compact fingerprint.

## 3. Compare matrix multiplication

Select **Matrix multiply**, use a `64 × 64` matrix, and run again. The CPU
reference and WGSL code both use the same conventional row-by-column multiply.
The GPU dispatch rounds each axis up to an `8 × 8` workgroup while the shader's
bounds check protects the logical matrix edge.

Try `16`, `64`, and `128`. Do not compare the timing with unrelated devices:
the browser, driver, power mode, and readback cost all matter.

## 4. Explore image convolution

Select **Image convolution** and switch among:

- Gaussian blur;
- sharpen; and
- edge detect.

The input image is generated deterministically. Pixel coordinates outside an
edge are clamped to the nearest valid coordinate on both CPU and GPU paths.
RGB results are clamped to `[0, 1]`; alpha is preserved.

## 5. Verify fallback explicitly

Choose **CPU reference only** and run any experiment. The result identifies the
CPU backend, reports `CPU loop` instead of a dispatch, and still performs
warmups, multiple samples, output validation, and export.

## 6. Export evidence

After a successful run, choose **Export JSON**. The versioned document contains
the parameters, backend, statistical summary, validation metrics, operation
model, byte counts, and output preview. It does not include personal
information, full arrays, or adapter identity.
