# PagedAttention

## The Problem

Traditional LLM inference allocates a **contiguous block of GPU memory** per sequence to store its KV cache. This causes two problems:

1. **Fragmentation** — memory is reserved upfront based on max sequence length, most of which sits unused
2. **No sharing** — even when two sequences share a common prefix (e.g. the same system prompt), their KV caches are stored separately

## The Idea: Virtual Memory for KV Cache

PagedAttention is inspired by how operating systems manage virtual memory with **paging**. Instead of one contiguous block per sequence, the KV cache is split into fixed-size **blocks** (pages), each holding the keys and values for a fixed number of tokens.

A **block table** maps each sequence's logical blocks to physical blocks in GPU memory — the same way an OS page table maps virtual pages to physical memory frames.

Physical blocks are allocated **on demand** as new tokens are generated, not upfront.

## Memory Efficiency

In traditional attention, any unused capacity within a pre-allocated contiguous buffer is wasted.  
In PagedAttention, waste only happens in the **last block** of each sequence (partially filled).

This results in **near-optimal memory utilization — under 4% waste**.

Less wasted memory → more sequences fit in GPU memory simultaneously → better GPU utilization → higher throughput.

## Memory Sharing (Copy-on-Write)

PagedAttention unlocks efficient memory sharing between sequences that share a common prefix.

**Example — parallel sampling**: generate N different completions for the same prompt. With PagedAttention:
- All N sequences map their prompt blocks to the **same physical blocks**
- Only diverging (newly generated) tokens get their own blocks
- Reference counts track how many sequences point to each physical block
- **Copy-on-Write** is used when a shared block needs to be modified

This cuts memory usage for parallel sampling and beam search by up to **55%**, translating to up to **2.2x improvement in throughput**.

## Summary

| Property | Traditional | PagedAttention |
|---|---|---|
| Memory layout | Contiguous per sequence | Non-contiguous blocks |
| Allocation | Upfront (max length) | On demand |
| Memory waste | High (fragmentation) | <4% (last block only) |
| Prefix sharing | Not possible | Yes (block table mapping) |
| Beam search overhead | High | Reduced by ~55% |
