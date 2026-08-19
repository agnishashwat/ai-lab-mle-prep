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

**Short answer:**

**Deeper:** (params + optimizer states + activations + KV cache; Adam = ~2 extra states)

**Watch out for:**

### Q: FSDP / ZeRO stages - what gets sharded and what are the tradeoffs?

**Short answer:**

**Deeper:**

**Watch out for:**

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
