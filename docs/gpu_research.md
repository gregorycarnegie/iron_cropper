# WebGPU ONNX Inference — Research & Implementation Notes (Batteries‑Included)

> Goal: run the **entire** ONNX graph on the GPU in **pure Rust** using **wgpu + WGSL**, keeping tensors resident on device (no host round‑trips between layers). Target model available: `face_detection_yunet_2023mar_640.onnx`.

---

## 🔎 Quick Checklist (conceptual subtasks)

- **Parse & plan**: Load ONNX, build a minimal IR (nodes, tensors, shapes, attributes), topologically sorted.
- **Map ops → kernels**: Conv2D, BatchNorm, Activations (ReLU/Sigmoid/SiLU), plus utility ops (Concat/Reshape/Pool if required).
- **Make tensors resident**: `GpuTensor` API, buffer layouts, bind‑group layouts, and lifetime/ownership rules.
- **Dispatch discipline**: select workgroup sizes, grid tiling, shared‑memory use, and command encoding strategy.
- **Static vs dynamic shapes**: prefer static input resolution; add runtime shape inference path for flexibility.
- **Validation & perf**: parity tests vs. CPU; profile, then iterate (tiling, vector loads, FP16, fusion).

> Purpose of this update: consolidate research into a single, actionable design; align with current sources:  
> **Rust**: `tensor.rs`, `ops.rs`, `mod.rs` — **WGSL**: `conv2d.wgsl`, `batch_norm.wgsl`, `activation.wgsl` — **Model**: `face_detection_yunet_2023mar_640.onnx`.

---

## 1) Overview / Architecture

We implement a **custom WebGPU execution engine** for a subset of ONNX ops sufficient for YuNet‑style face detection. The engine has three layers:

1. **Model IR** (Rust): parsed ONNX graph → `Node { kind, inputs, outputs, attrs }`, `Tensor { name, dtype, shape, data? }`.
2. **Operator registry** (Rust): maps `NodeKind::Conv/BatchNormalization/Activation/...` → callable GPU pipeline wrappers.
3. **WGSL kernels**: one compute pipeline per op family; parametric via uniforms; all I/O in storage buffers.

### **Design principles**

- **GPU residency**: upload once (inputs, weights); execute all layers on device; download once at the end.
- **Zero-copy chaining**: `GpuTensor` carries `(device, buffer, shape)`; outputs feed directly into next op.
- **Single‑pass orchestration**: encode all dispatches for a graph execution into one command buffer when shapes are static.

---

## 2) ONNX Parsing & Minimal IR (Rust)

- Parse model into a compact IR. Options:
  - Use a light ONNX proto reader (prost‑generated) or a crate like `onnx-ir`; or
  - Implement a focused reader that extracts only: **initializers**, **inputs/outputs**, **nodes** (type + attrs), **topo order**.
- Extract **constant tensors** (weights/bias/BN params) and upload once to `GpuTensor`.
- Compute **shapes**:
  - Prefer **static** input (YuNet is typically fixed resolution). Precompute all intermediate shapes.
  - For dynamic inputs, run a small **shape inference**: conv output size from `stride/pad/dilate`, pooling formulas, etc.
- Build a **value table**: ONNX name → `GpuTensor` (for constants at init; for activations during execution).

### **IR sketch**

```rust
enum NodeKind { Conv, BatchNormalization, Relu, Sigmoid, SiLU, Add, Mul, Concat, Reshape, PoolMax, PoolAvg, /* … */ }

struct Node {
    kind: NodeKind,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: Attrs, // strides, pads, dilations, groups, axis, etc.
}

struct TensorInit {
    name: String, dtype: DType, shape: Vec<usize>, bytes: Vec<u8> // weight blobs
}
```

---

## 3) GPU Resident Tensors & Memory Layout

**`GpuTensor`** (Rust)

- Owns a `wgpu::Buffer`, element `DType` (start with `f32`), and `Shape { n,c,h,w }` (NCHW by default).
- Provides: `from_slice(&[f32])`, `uninitialized(shape)`, `download()` (tests/IO), and view helpers (e.g., channel strides).

### **Layout choice**

- Start with **NCHW** to match many ONNX CV models and current WGSL kernels.
- Keep a **layout flag** in `GpuTensor` to future‑proof for NHWC. Kernels read dimensions from uniforms; index math branches by layout if/when needed.

### **Bind group conventions**

- Binding 0..N: storage buffers (inputs/weights/params/output). Last binding: a small **uniform** struct with dims & hyper‑params.
- One **pipeline** per op family; multiple **bind groups** per invocation.

---

## 4) WGSL Kernels (current & planned)

### 4.1 Conv2D (`conv2d.wgsl`)

- Workgroup: start at **8×8×1** (64 threads). Each thread computes one output pixel for a single `out_channel` tile.
- Uniforms: `{ in_w, in_h, in_c, out_w, out_h, out_c, k_w, k_h, stride_x, stride_y, pad_x, pad_y, groups }`.
- Indexing: compute source window origin; loop k_h/k_w and `in_c/group` to accumulate FMA into register, add bias, write.
- **Roadmap optimizations**:
  - Shared‐memory tile of input patch (`var<workgroup>`), cooperative loads + `workgroupBarrier()`.
  - Vectorized loads (4× `f32`) where alignment allows.
  - **Fusion** hooks: optionally apply BN/activation before writing out (see §7).

### 4.2 Batch Normalization (`batch_norm.wgsl`)

- Thread‑per‑element; z‑dimension over `channel`, xy over spatial.
- Read `gamma,beta,mean,var` by channel; compute `(x − mean) * rsqrt(var+eps) * gamma + beta` into out buffer.

### 4.3 Activations (`activation.wgsl`)

- Thread‑per‑element (`@workgroup_size(256)`) with a `mode` uniform: 0=ReLU, 1=Sigmoid, 2=SiLU, etc.
- **In‑place** allowed for bandwidth savings when safe.

### 4.4 Utilities (as needed by model)

- Pooling (Max/Avg), Concat, Reshape (meta; no kernel), Slice, Transpose (if layout bridging).
- Post‑proc kernels (e.g., decode head tensors to boxes/scores; optional GPU NMS).

---

## 5) Execution Pipeline (Rust `ops.rs` / `mod.rs`)

1. **Initialize**: create `GpuContext` (adapter/device/queue); compile pipelines once; upload **initializers** to `GpuTensor`.
2. **Preprocess**: convert image → tensor (normalize, mean/std, channel order, resize to model resolution, e.g., 640 on YuNet).
3. **Execute graph** (topo order):
   - Look up input tensors from value table.
   - Call registry wrapper (e.g., `conv2d_tensor(&x, &w, &b, &cfg)`), producing a new `GpuTensor`.
   - Store outputs back into value table; chain into next op **without download**.
4. **Postprocess**: last layer(s) to CPU only if needed (e.g., decode & NMS on host initially; later move to GPU).
5. **Command encoding**: for **static shapes**, pre‑record in a single `CommandEncoder` and reuse per inference (“graph capture” concept).

### **Dispatch discipline**

- Compute `dispatch = (ceil(out_w/8), ceil(out_h/8), out_c)` for Conv/BN; or 1D `ceil(nelems/256)` for elementwise.
- Ensure total threads ≤ hardware limits and keep groups large enough to saturate SMs.

---

## 6) Static vs. Dynamic Shapes

- **Static** (recommended): lock input shape (YuNet model file suggests 640‑side). Advantages:
  - Pre‑allocate all intermediate `GpuTensor`s; reuse across runs.
  - Pre‑build one command buffer per pass; minimal per‑run CPU work.
  - Easier fusion and kernel specialization.
- **Dynamic** (supported path):
  - Lightweight **shape inference** per op at runtime; allocate outputs on demand (with a shape→buffer cache).
  - Recompute dispatch sizes from uniforms; reuse pipelines (no shader recompile).
  - Watch allocator churn; consider pooling to avoid fragmentation.

---

## 7) Performance Notes & Roadmap

- **Tiling + shared memory** in Conv2D for higher arithmetic intensity.
- **Vectorization** (4× loads/stores) and loop unrolling for inner kernel loops.
- **Fusion opportunities**: Conv2D + Bias + BatchNorm + Activation in a single kernel reduces memory traffic.
- **Precision**: optional FP16 path (weights & activations) with accumulate in f32 if needed for stability.
- **Memory planning**: static lifetime analysis to reuse large buffers across non‑overlapping layers.
- **IO binding**: keep pre/post on GPU in native or WASM targets where feasible.

---

## 8) Model‑Specific: YuNet Face Detection

- **Input**: BGR/RGB to model layout (confirm channel order), resize/pad to **640** on long side (keep aspect or letterbox to match training).
- **Normalization**: model‑specific mean/std or scale to [0,1].
- **Heads**: the detection head typically outputs **boxes + scores + landmarks** at multiple strides; implement GPU decode (anchors, grid) to minimise CPU work.
- **NMS**: start on CPU for simplicity; later port to WGSL (bitmask/warp‑like reduction).

---

## 9) Testing & Validation

- **Unit tests**:
  - Operator parity: Conv/BN/Activation vs. CPU reference (within ε).
  - End‑to‑end: fixed input → compare final tensors to known outputs.
- **Numerical**:
  - Watch for off‑by‑one in padding; confirm dilations/groups; clamp index bounds in WGSL.
- **Profiling**:
  - Capture GPU timing per kernel; iterate workgroup size, tile size, and fusion.

---

## 10) Minimal Public API (sketch)

```rust
// one‑time
let mut engine = GpuEngine::new(device, queue)?;
engine.load_onnx("face_detection_yunet_2023mar_640.onnx")?;

// per‑inference
let x = engine.upload_image(&img_preprocessed)?; // -> GpuTensor NCHW
let y = engine.run(&x)?;                         // runs full graph on GPU, returns output GpuTensor
let out = y.download()?;                         // if CPU needed
```

Internals: `GpuEngine` owns pipelines, a value table, and an op registry. `run()` iterates topo nodes and dispatches kernels with proper binds/dispatch sizes.

---

## 11) File Map (current repo)

- **Rust**: `tensor.rs` (GpuTensor + shapes), `ops.rs` (GPU wrappers + dispatch), `mod.rs` (context + pipelines).
- **WGSL**: `conv2d.wgsl`, `batch_norm.wgsl`, `activation.wgsl`.
- **Model**: `face_detection_yunet_2023mar_640.onnx` (YuNet).

---

## 12) Next Actions (implementation‑ready)

1. **Finalize IR** and ONNX loader for required ops; precompute static shapes for 640‑input.
2. **Harden kernels**: bounds checks, bias add, eps in BN, activation modes.
3. **End‑to‑end path**: encode one command buffer for full graph; confirm GPU‑only chaining.
4. **Parity tests**: compare against CPU ORT/NumPy for 2–3 images; assert max|Δ| < 1e‑4.
5. **Perf pass**: try 8×8 vs 16×8 tiles; add shared‑mem tiling in Conv; measure speedup.
6. **Optional**: add GPU decode + CPU NMS; then port NMS to WGSL.

---

*Maintainers’ note:* This document supersedes prior notes and should be kept in sync with code in `tensor.rs`, `ops.rs`, `conv2d.wgsl`, `batch_norm.wgsl`, and `activation.wgsl`. Update the **Next Actions** checklist as milestones land.
