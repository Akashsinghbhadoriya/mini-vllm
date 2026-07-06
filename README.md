# miniVLLM

A minimal from-scratch implementation of the core ideas behind [vLLM](https://github.com/vllm-project/vllm) — built for learning and understanding how modern LLM inference engines work.

## What This Is

vLLM achieves up to 24x better throughput than naive HuggingFace inference by rethinking how GPU memory is managed during generation. This project implements the foundational concepts:

- **Request lifecycle** with explicit status transitions (STARTED → WAITING → RUNNING → FINISHED / FAILED)
- **Prefill phase** — processes the full prompt batch together, builds the initial KV cache per request
- **Decode phase** — generates one token at a time across all active requests, reusing cached KV values
- **Scheduler** — manages the active request batch, enforces a max batch size, and drives status transitions
- **Engine loop** — orchestrates prefill + decode in two modes: static batching and continuous batching
- **Continuous batching** — a background engine thread pulls from a request queue and dynamically adds new requests mid-flight, without waiting for the current batch to finish
- **Custom attention** — multi-head attention with RoPE and GQA support, replacing the HuggingFace attention layer
- **Paged attention** — block-based KV cache management inspired by vLLM's PagedAttention
- **KV cache manager** — global memory pool of fixed-size blocks, allocated per-request and freed on completion
- **Server / Client** — thread-safe server that accepts concurrent client requests and tracks per-request latency

## Project Structure

```
miniVllm/
├── main.py                    # Entry point — spawns concurrent client threads against the Server
├── client.py                  # Client: submits a prompt to the server and prints the response + latency
├── benchmark.py               # Benchmarks sequential vs batch vs continuous batching throughput
├── inspect_forward.py         # Scratch file for exploring model forward pass outputs
│
├── core/
│   ├── server.py              # Server: wraps Engine with a thread-safe submit interface + latency tracking
│   ├── engine.py              # Core inference engine (static batch mode + continuous batching serve loop)
│   ├── model_runner.py        # Wraps LlamaModel: prefill_batch() and decode_batch()
│   └── scheduler.py           # Manages active request batch, capacity, and status transitions
│
├── request/
│   ├── request.py             # Request dataclass + RequestStatus enum
│   ├── request_queue.py       # Thread-safe inbound request queue (deque + lock)
│   └── response_queue.py      # Thread-safe outbound response queue (deque + lock)
│
├── attention/
│   ├── rope.py                # Rotary Position Embeddings (RoPE)
│   ├── attention.py           # Custom multi-head attention with KV cache support
│   └── paged_attention.py     # Paged attention with block-table-based KV gather/write
│
├── kv_cache/
│   ├── memory_block.py        # Fixed-size KV block (single unit of preallocated memory)
│   ├── block_table.py         # Per-request mapping: logical sequence → physical blocks
│   └── kv_cache_manager.py    # Global block pool: allocate, free, track stats
│
├── models/
│   ├── llama_model.py         # Custom LlamaForCausalLM wrapper with pluggable attention
│   └── llama_decoder.py       # Single transformer layer: pre-norm attention + MLP
│
├── docs/
│   ├── vllm.md                # Notes on vLLM concepts: prefill vs decode
│   └── PagedAttention.md      # Notes on PagedAttention memory management
│
└── requirements.txt           # torch, transformers
```

## How It Works

### 1. Request

Each `Request` holds:
- The prompt string
- Token IDs generated so far
- The last generated token (used as input for the next decode step)
- A `kv_cache` list of `(k, v)` tuples per layer
- A `kv_seq_len` tracking how many tokens are currently cached
- A `block_table` for paged memory management
- A `RequestStatus` tracking pipeline position
- A `threading.Event` (`completed`) that blocks the client until the response is ready
- `start_time` / `end_time` for per-request latency measurement

Status transitions:

```
STARTED → WAITING → RUNNING → FINISHED
                             ↘ FAILED
```

### 2. Scheduler

The `Scheduler` manages the active request batch with a configurable `max_batch` size:
- `add_active(requests)` — adds requests to the active batch
- `get_active()` — returns all currently active requests
- `remove_finished()` — evicts finished requests and returns them
- `has_capacity()` / `get_capacity()` — checks how many slots are free in the batch
- `has_active()` — returns whether any requests are currently running

### 3. Engine — Two Modes

**Static Batch Mode** (`generate` / `generate_batch`):

```
add_requests(reqs) → scheduler manages batch
     ↓
prefill_batch(all_requests)  →  mark all RUNNING
     ↓
while has_pending():
    decode_batch(active_requests)  →  check stop conditions per request
    scheduler.remove_finished()
     ↓
all requests FINISHED
```

All requests are prefilled together in a single batched forward pass. The decode loop then processes all active requests jointly each step, reusing each request's `kv_cache`.

**Continuous Batching Mode** (`serve`, used by Server):

```
Engine runs in a background thread (start() / stop())
     ↓
loop:
    if scheduler has capacity:
        dequeue up to N new requests from request_queue
        prefill_batch(new_requests)         ← adds to active batch mid-flight
        kv_manager.allocate_for_request()   ← reserve blocks for each new request
    if scheduler has active requests:
        decode_batch(active_batch)
        finished = scheduler.remove_finished()
        for each finished:
            kv_manager.free_request()       ← return blocks to pool
            decode_text(request)            ← decode token IDs → string
            response_queue.enqueue(request) ← unblocks waiting client thread
```

New requests are absorbed into the running batch as soon as capacity opens up — no waiting for the current batch to drain first.

### 4. Server + Client

**Server** wraps the engine and exposes `submit_request(prompt)`:
- Assigns a thread-safe incrementing request ID
- Enqueues the request and blocks on `request.completed` (a `threading.Event`)
- Returns `(generated_text, request_id, latency)` once the engine signals completion

**Client** calls `server.submit_request(prompt)` from its own thread and prints the response with latency. Multiple clients can submit concurrently — the engine batches them automatically.

### 5. ModelRunner

Wraps `LlamaModel` (custom wrapper over HuggingFace) with two batch-oriented methods:

- **`prefill_batch(requests)`** — tokenizes and left-pads the prompt batch, computes position IDs from the attention mask, runs one joint forward pass, extracts per-request KV caches from the batched output, and samples the first token for each request
- **`decode_batch(requests)`** — stacks the last token from each request into a `[B, 1]` tensor, pads all KV caches to the max sequence length, runs a forward pass, samples the next token, and updates each request's KV cache and `kv_seq_len`

Sampling uses softmax + multinomial sampling (temperature=1).

### 6. Custom Attention

The HuggingFace attention layer is replaced with a custom implementation that exposes explicit KV cache control.

**`attention/rope.py`** — Rotary Position Embeddings:
- `apply_rotary_emb(q, k, cos, sin)` applies the RoPE rotation to query and key tensors, enabling position-aware attention without additive position embeddings

**`attention/attention.py`** — `Attention(nn.Module)`:
1. Projects hidden states → Q, K, V
2. Reshapes to `(B, H, S, D)`
3. Applies RoPE
4. Concatenates cached KV from previous steps (if present)
5. Handles GQA (Grouped Query Attention) by repeating KV heads
6. Runs `scaled_dot_product_attention` — causal during prefill, non-causal during decode
7. Returns attention output and `(new_k, new_v)` for caching

**`attention/paged_attention.py`** — `PagedAttention(Attention)`:
- Extends standard attention to operate on block-table-based memory
- `_gather_from_blocks(block_table, kv_seq_len)` — reconstructs the KV cache from non-contiguous physical blocks
- `_write_to_blocks(block_table, k, v, start_pos)` — writes new KV tokens into blocks, allocating tensor storage on first write
- During decode: gathers past KV from blocks, concatenates with current tokens, then writes back

### 7. KV Cache Management

**`kv_cache/memory_block.py`** — `MemoryBlock`:
- A fixed-size unit of preallocated memory (`capacity` tokens)
- Tracks `used_tokens`, `is_allocated`, and holds `k_cache` / `v_cache` tensors
- `has_space()`, `remaining_capacity()`, `add_tokens()`, `clear()`

**`kv_cache/block_table.py`** — `BlockTable`:
- Per-request page table: a list of `MemoryBlock` objects assigned to one request
- `append(block)`, `last_block()`, `total_capacity()`, `used_tokens()`
- Enables non-contiguous memory — each request references a subset of global blocks

**`kv_cache/kv_cache_manager.py`** — `KVCacheManager`:
- Manages a global pool of `num_blocks` blocks (default 1024, size 16 tokens each)
- `allocate_for_request(block_table, num_tokens)` — fills the last block first, allocates new blocks when needed
- `free_request(block_table)` — returns all blocks from a finished request back to the free pool
- `stats()` — returns total / allocated / free block counts

### 8. Custom LlamaModel

**`models/llama_decoder.py`** — `LlamaDecoderLayer(nn.Module)`:
- Single transformer layer with pluggable attention backend
- Copies layer norms and MLP from the HuggingFace layer
- Forward: pre-norm → attention → residual → post-norm → MLP → residual
- Returns `(hidden_states, new_kv)`

**`models/llama_model.py`** — `LlamaModel(nn.Module)`:
- Extracts `embed_tokens`, `norm`, `lm_head`, and `rotary_emb` from HuggingFace
- Builds a list of `LlamaDecoderLayer` wrappers, each using `PagedAttention`
- Forward: embed → N decoder layers with KV cache → norm → logits
- Returns `(logits, new_kv_caches)` — fully custom forward pass

## Architecture Overview

```
Client Threads (main.py)
        │ submit_request(prompt)
        ▼
┌───────────────────┐
│   Server          │  Creates Request, blocks on request.completed
└───────┬───────────┘
        │ enqueue
        ▼
┌───────────────────┐
│   RequestQueue    │  Thread-safe deque
└───────┬───────────┘
        │ dequeue_many()
        ▼
┌─────────────────────────────────────────┐
│   Engine (daemon thread)                │
│                                         │
│  ┌─────────────┐   ┌──────────────────┐ │
│  │  Scheduler  │   │  KVCacheManager  │ │
│  │ (batch=8)   │   │  (block pool)    │ │
│  └──────┬──────┘   └────────┬─────────┘ │
│         │                   │           │
│  ┌──────▼───────────────────▼────────┐  │
│  │   Engine.serve()                  │  │
│  │   prefill_batch() / decode_batch()│  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │   ModelRunner                     │  │
│  │   ┌─────────────────────────────┐ │  │
│  │   │   LlamaModel                │ │  │
│  │   │   ├── embed_tokens          │ │  │
│  │   │   ├── LlamaDecoderLayer x N │ │  │
│  │   │   │   ├── LayerNorm         │ │  │
│  │   │   │   ├── PagedAttention    │ │  │
│  │   │   │   │   └── RoPE          │ │  │
│  │   │   │   └── MLP               │ │  │
│  │   │   ├── final norm            │ │  │
│  │   │   └── lm_head               │ │  │
│  │   └─────────────────────────────┘ │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
        │ request.completed.set()
        ▼
┌───────────────────┐
│   ResponseQueue   │  Finished requests
└───────┬───────────┘
        │ unblocks client thread
        ▼
Client prints (text, latency)
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

> Requires access to `meta-llama/Llama-3.2-3B` on HuggingFace. Run `huggingface-cli login` first or set `HUGGING_FACE_HUB_TOKEN`.

`main.py` starts a Server and submits three prompts concurrently from separate client threads.

To use a lighter model for testing:

```python
# in server.py or model_runner.py
model_runner = ModelRunner(model_name="gpt2")
```

## Benchmarking

`benchmark.py` measures throughput across three modes:

```bash
python benchmark.py
```

| Mode | Description |
|---|---|
| Sequential | Each request prefilled and decoded independently, one after another |
| Batch | All requests prefilled together in one pass, decoded together each step |
| Continuous Batching | Requests submitted concurrently via client threads; engine absorbs them as capacity opens |

Example output:
```
============================
Mini-vLLM Benchmark
============================
Requests: 5
Model: meta-llama/Llama-3.2-3B

Sequential
Time: 142.3 s

Batch
Time: 38.7 s

Continuous Batching
Time: 35.1 s

Speedup (sequential → batch): 3.67x
```

Batching wins by amortizing the high compute cost of prefill across multiple sequences and making better use of GPU memory bandwidth during decode. Continuous batching further improves utilization by never leaving the GPU idle while new requests are waiting.

## Key Concepts

See [`docs/vllm.md`](./docs/vllm.md) for notes on the prefill vs decode distinction, and [`docs/PagedAttention.md`](./docs/PagedAttention.md) for how vLLM extends this with paged memory management.

## What's Missing (vs Real vLLM)

This is intentionally minimal. Real vLLM adds:

| Feature | Status |
|---|---|
| PagedAttention (non-contiguous KV blocks) | Implemented (architecture + block manager; decode still uses dense tensor KV) |
| Custom attention with RoPE + GQA | Implemented |
| Continuous batching | Implemented |
| Async engine / API server | Partial (threaded Server) |
| GPU memory pre-allocation | Not implemented |
| Beam search / parallel sampling | Not implemented |
| Quantization (AWQ, GPTQ) | Not implemented |
| Preemption / request eviction | Not implemented |

The goal is to understand the **prefill/decode split**, **KV cache reuse**, **continuous batching**, and **paged memory management** — the core ideas everything else in vLLM is built on top of.
