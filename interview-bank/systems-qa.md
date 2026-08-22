# Interview Bank - ML Systems Q&A

## Format

### Q: <question>

**Short answer:** <one or two sentences>

**Deeper:** <the fuller explanation, with numbers where possible>

**Watch out for:** <the follow-up / gotcha>

---

### Q: A 7B BF16 model is only 14 GB, so it fits comfortably on a 16 GiB GPU. Correct?

**Short answer:** Not comfortably at batch 1 and 4k context under the stated
MHA assumptions. The weights are about 13.0 GiB and the KV cache is another
2 GiB, already leaving too little operational headroom.

**Deeper:** The cache is
`2 * 32 layers * 32 KV heads * 128 head dimension * 1 * 4096 * 2 bytes`,
which is exactly 2 GiB. Adding that to the weights leaves only about 1 GiB for
runtime state, workspaces, temporary tensors, and allocator fragmentation.

**Watch out for:** This is scoped, not universal. Shorter context, GQA/MQA, or
quantized weights can change the fit. Keep GB and GiB consistent.

---

### Q: Can a 7B model be fully fine-tuned with mixed-precision Adam on one 80 GiB GPU?

**Short answer:** No, not with the classic unsharded layout. Its persistent
model state alone is 104.3 GiB, before activations and overhead.

**Deeper:** BF16 weights + BF16 gradients + FP32 master weights + two FP32 Adam
moments cost `2 + 2 + 4 + 8 = 16 bytes/parameter`. For 7B parameters that is
112 GB, or 104.3 GiB. Saved activations, temporary kernel workspaces,
runtime/allocator overhead, and possibly communication buffers must still be
added.

**Watch out for:** Do not add a serving KV cache as a separate training bucket;
training K/V tensors are saved activations. Different optimizer layouts,
sharding, offload, checkpointing, or parameter-efficient tuning change the
answer.

---

### Q: Why can two implementations of the same model and batch have different peak activation memory?

**Short answer:** Activation memory depends on implementation and peak tensor
liveness, not only model architecture and input shape.

**Deeper:** FlashAttention tiles the computation instead of materializing full
`T*T` attention matrices. Gradient checkpointing discards selected saved
tensors and recomputes them. Fusion, activation precision, padding, and which
tensors are simultaneously live also change the peak.

**Watch out for:** Weight quantization alone does not reduce saved activation
memory. An activation estimate must state the kernel, checkpointing policy,
dtype, and what the estimate assumes is live at once.

---

### Q: Estimate the memory to fine-tune a 7B model (full FT vs LoRA).

**Short answer:** Under the classic mixed-precision Adam assumptions, full FT
needs `16 bytes/parameter`, so 7B parameters need 112 GB or 104.31 GiB before
activations. For rank-16 LoRA on only `q_proj` and `v_proj` in 32
hidden-size-4096 layers, 8,388,608 trainable adapter parameters put persistent
state at about 13.164 GiB. An idealized 4-bit QLoRA version is about 3.385 GiB.

**Deeper:** The LoRA adapter count is
`32 * 2 * 16 * (4096 + 4096) = 8,388,608`. Its classic trainable
state is exactly 0.125 GiB; the frozen BF16 base is 13.0385 GiB. QLoRA
changes the frozen-base term from 2 bytes/parameter to an idealized 0.5
bytes/parameter while retaining the adapter state.

**Watch out for:** These are persistent-state estimates, not peak-memory
estimates. Saved activations, temporary and dequantization buffers, quantization
metadata, communication buffers, allocator/runtime headroom, and kernel details
are omitted. Full FT has a larger update parameterization, but that does not
guarantee better held-out quality.

---

### Q: FSDP / ZeRO stages - what gets sharded and what are the tradeoffs?

**Short answer:** ZeRO-1 shards FP32 master weights and optimizer moments;
ZeRO-2 additionally shards gradients; ZeRO-3 additionally shards parameters.
Under a `2 + 2 + 4 + 8` byte layout and world size `N`, the ideal persistent
per-rank formulas are `P(4 + 12/N)`, `P(2 + 14/N)`, and `16P/N`.

**Deeper:** ZeRO-3/FSDP all-gathers the current wrapped unit's parameters for
compute and reduce-scatters gradients. Peak memory therefore includes the full
current unit, possibly a prefetched next unit, communication buffers,
activations, temporary tensors, and runtime headroom in addition to persistent
local shards.

**Watch out for:** `16P/N` is not peak memory. Higher sharding stages purchase
memory with parameter communication and orchestration overhead; FSDP wrapping
and prefetch policies change both transient materialization and throughput.

---

### Q: A QLoRA run fits at short context but OOMs when sequence length doubles. Diagnose it.

**Short answer:** The growing bucket is most likely saved activations, which
QLoRA does not quantize. If full attention matrices are materialized, their
storage scales as `T^2`; with FlashAttention, residual/hidden activation storage
is approximately linear in `T`.

**Deeper:** For naive attention, use an efficient-attention kernel first. If
FlashAttention is already active, use gradient checkpointing or a smaller
microbatch. Checkpointing reduces saved activation liveness across layers; it
does not change the sequence-length exponent.

**Watch out for:** QLoRA changes the frozen base from BF16's 2 bytes/parameter
to an idealized 0.5 bytes/parameter and reduces trainable state through LoRA.
It does not make activations, adapter state, dequantization buffers, or runtime
overhead 4-bit.

---

### Q: How does continuous batching + paged attention improve inference throughput?

**Short answer:**

**Deeper:**

**Watch out for:**

### Q: Data vs model vs pipeline parallelism - when do you reach for each?

**Short answer:**

**Deeper:**

**Watch out for:**

### Q: Design an end-to-end SFT + DPO + eval pipeline. Where are the bottlenecks?

**Short answer:**

**Deeper:**

**Watch out for:**
