# Interview Bank - ML Systems Q&A

## Format

### Q: <question>

**Short answer:** <one or two sentences>

**Deeper:** <the fuller explanation, with numbers where possible>

**Watch out for:** <the follow-up / gotcha>

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
