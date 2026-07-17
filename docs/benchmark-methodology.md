# Benchmark methodology

## Scope

WebGPU Kernel Lab measures an educational end-to-end browser path. It is useful
for comparing settings during one session and for understanding the cost
components surrounding a compute shader. It is not designed to rank hardware.

## GPU sample boundary

Each GPU sample:

1. reuses a cached compute pipeline;
2. creates and fills fresh storage and uniform buffers;
3. creates an output and mapped readback buffer;
4. encodes the compute pass and copy;
5. starts the timer immediately before queue submission;
6. submits the command buffer;
7. awaits `mapAsync(GPUMapMode.READ)`;
8. stops the timer; and
9. copies the mapped data into a standalone `Float32Array`.

Buffer creation and encoding occur before the timer. Queue submission, GPU
execution, output copy, and map/readback wait are included.

## CPU sample boundary

Each CPU sample starts immediately before the typed reference function and ends
after it returns its new `Float32Array`. A zero-delay yield follows each sample
so the browser can update progress between longer runs.

## Statistics

Warmup results are discarded. Measured samples produce:

- minimum and maximum;
- arithmetic mean;
- linearly interpolated median and p95; and
- population standard deviation.

Throughput divides a documented modeled operation count by median duration. The
model is:

- vector add: one operation per element;
- matrix multiply: `2 × N³` operations; and
- convolution: `width × height × 3 channels × 9 taps × 2` operations.

These counts are explanatory models, not an assertion about the exact
instructions executed by a compiler or device.

## Reproducibility guidance

For a more stable comparison:

- keep the tab visible;
- use the same browser and power mode;
- close unrelated heavy workloads;
- run more than five measured samples; and
- compare medians and distributions, not a single fastest number.
