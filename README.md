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
- **Prefix caching** — LRU cache of completed KV blocks matched by token prefix, avoids redundant prefill compute for shared prompt prefixes
- **OpenAI-compatible REST API** — `/v1/completions`, `/v1/chat/completions`, and `/metrics` endpoints via FastAPI
- **Streaming** — SSE token streaming with per-chunk inference metrics
- **Visualization dashboard** — multi-user Streamlit UI with real-time streaming and per-request metrics
- **Server / Client** — thread-safe server that accepts concurrent client requests and tracks per-request latency

## Project Structure

```
miniVllm/
├── main.py                    # Entry point — spawns concurrent client threads against the Server
├── client.py                  # Client: submits a prompt to the server and prints the response + latency
├── benchmark.py               # Benchmarks sequential vs batch vs continuous batching throughput
├── ui.py                      # Streamlit multi-user inference dashboard (requires API running)
│
├── core/
│   ├── server.py              # Server: wraps Engine with a thread-safe submit interface + latency tracking
│   ├── engine.py              # Core inference engine (static batch mode + continuous batching serve loop)
│   ├── model_runner.py        # Wraps LlamaModel: prefill_batch() and decode_batch()
│   └── scheduler.py           # Manages active request batch, capacity, and status transitions
│
├── request/
│   ├── request.py             # Request dataclass + RequestStatus enum
│   ├── handle.py              # Async/streaming request handle with token-by-token yield
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
│   ├── kv_cache_manager.py    # Global block pool: allocate, free, prefix lookup, track stats
│   └── prefix_cache.py        # LRU cache of completed KV blocks keyed by token prefix hash
│
├── models/
│   ├── llama_model.py         # Custom LlamaForCausalLM wrapper with pluggable attention
│   └── llama_decoder.py       # Single transformer layer: pre-norm attention + MLP
│
├── api/
│   ├── app.py                 # FastAPI app with CORS middleware
│   ├── routes.py              # /v1/completions, /v1/chat/completions, /metrics endpoints
│   ├── schemas.py             # Pydantic request/response models (completion, chat, streaming)
│   └── stream.py              # SSE streaming generator with per-chunk token + metrics events
│
├── docs/
│   ├── vllm.md                # Notes on vLLM concepts: prefill vs decode
│   └── PagedAttention.md      # Notes on PagedAttention memory management
│
└── requirements.txt           # torch, transformers, fastapi, uvicorn, streamlit, pydantic, requests
```

## Quickstart

### 1. Install dependencies

```bash
git clone <repo-url>
cd miniVllm
pip install -r requirements.txt
```

### 2. Model access

The default model is `meta-llama/Llama-3.2-3B`. You need a HuggingFace account with access granted:

```bash
huggingface-cli login
```

To skip this and use a lightweight model for quick testing, change the model name in `core/server.py`:

```python
model_runner = ModelRunner(model_name="gpt2")
```

### 3a. Run a direct concurrent inference test

```bash
python main.py
```

Spawns three client threads submitting prompts concurrently. The engine batches them automatically.

### 3b. Run the REST API + Visualization Dashboard

Open two terminals:

**Terminal 1 — start the API server:**

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — start the Streamlit dashboard:**

```bash
streamlit run ui.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Visualization Dashboard

`ui.py` is a multi-user inference monitoring interface built with Streamlit.

**Features:**
- Up to 8 concurrent user panels in a 2-column grid layout
- Add / remove user panels dynamically with the "+ Add User" button
- Each panel has an independent prompt input, max tokens slider, and Generate button
- Tokens stream in real-time with a cursor animation (`▌`) as the engine generates
- Per-request metrics displayed after generation completes:

| Metric | Description |
|---|---|
| Tokens/sec | Generation throughput for this request |
| TTFT | Time to first token (ms) |
| Prefix cache | Hit or miss — whether KV blocks were reused from a prior request |
| KV blocks | Number of memory blocks allocated for this request |
| Batch ID | Which engine batch this request was processed in |
| Total latency | End-to-end wall time (ms) |

- Global metrics bar (refreshes every 10 seconds): active requests, queue size, total requests served
- The dashboard connects to the API server at `localhost:8000` — start the API first

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
- `lookup_prefix(token_ids)` — checks the prefix cache for reusable KV blocks matching the request's prompt
- `cache_completed_blocks(token_ids, block_table)` — stores fully filled blocks into the prefix cache after prefill
- `stats()` — returns total / allocated / free block counts

### 8. Prefix Caching

**`kv_cache/prefix_cache.py`** — `PrefixCache`:
- An LRU `OrderedDict` keyed by the hash of a token prefix
- `lookup(token_ids)` — returns the longest matching prefix and its cached blocks (if any)
- `insert(token_ids, blocks)` — stores completed blocks; evicts least-recently-used entries when capacity is exceeded
- Reference-counted blocks — shared prefix blocks are not freed until all referencing requests finish

On each new request, `KVCacheManager.lookup_prefix()` walks the token sequence in block-sized chunks and returns any cached blocks that match. The prefill phase skips those tokens entirely, starting the KV computation from the first uncached position. This is especially effective when many requests share a long system prompt.

### 9. REST API & Streaming

**`api/app.py`** — FastAPI application with CORS middleware. Initializes a shared `Server` instance on startup.

**`api/routes.py`** — Three endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/completions` | POST | Text completion (streaming or blocking) |
| `/v1/chat/completions` | POST | Chat with message history (streaming or blocking) |
| `/metrics` | GET | Global engine stats: active requests, queue size, total served |

**`api/schemas.py`** — Pydantic models for request/response shapes, compatible with the OpenAI API format.

**`api/stream.py`** — SSE streaming generator:
- Polls the request handle token by token
- Yields each token as a `data: {...}` SSE event
- Emits a final `data: {"type": "metrics", ...}` event containing TTFT, tokens/sec, prefix cache status, KV blocks used, and total latency

### 10. Custom LlamaModel

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
Client Threads (main.py)             HTTP Clients / Dashboard (ui.py)
        │ submit_request(prompt)              │ POST /v1/completions
        ▼                                     ▼
┌───────────────────┐             ┌───────────────────────┐
│   Server          │             │   FastAPI (api/)       │
│ (thread-safe)     │             │ routes + SSE streaming │
└───────┬───────────┘             └──────────┬────────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       │ enqueue Request
                       ▼
              ┌───────────────────┐
              │   RequestQueue    │  Thread-safe deque
              └───────┬───────────┘
                      │ dequeue_many()
                      ▼
┌─────────────────────────────────────────────────────────┐
│   Engine (daemon thread)                                │
│                                                         │
│  ┌─────────────┐   ┌──────────────┐  ┌───────────────┐ │
│  │  Scheduler  │   │ KVCache      │  │ PrefixCache   │ │
│  │ (batch=8)   │   │ Manager      │  │ (LRU blocks)  │ │
│  └──────┬──────┘   └──────┬───────┘  └───────┬───────┘ │
│         │                 │                   │         │
│  ┌──────▼─────────────────▼───────────────────▼──────┐  │
│  │   Engine.serve()                                   │  │
│  │   lookup_prefix() → prefill_batch() / decode_batch│  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │   ModelRunner                                      │  │
│  │   ┌─────────────────────────────────────────────┐ │  │
│  │   │   LlamaModel                                │ │  │
│  │   │   ├── embed_tokens                          │ │  │
│  │   │   ├── LlamaDecoderLayer x N                 │ │  │
│  │   │   │   ├── LayerNorm                         │ │  │
│  │   │   │   ├── PagedAttention                    │ │  │
│  │   │   │   │   └── RoPE                          │ │  │
│  │   │   │   └── MLP                               │ │  │
│  │   │   ├── final norm                            │ │  │
│  │   │   └── lm_head                               │ │  │
│  │   └─────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
        │ request.completed.set() / SSE stream
        ▼
Client prints (text, latency) / Dashboard renders tokens + metrics
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

## What's Implemented vs What's Missing

| Feature | Status |
|---|---|
| Paged attention (non-contiguous KV blocks) | Implemented |
| Custom attention with RoPE + GQA | Implemented |
| Continuous batching | Implemented |
| Prefix caching (LRU, block-level) | Implemented |
| OpenAI-compatible API server | Implemented |
| SSE token streaming with metrics | Implemented |
| Multi-user visualization dashboard | Implemented |
| GPU memory pre-allocation | Not implemented |
| Beam search / parallel sampling | Not implemented |
| Quantization (AWQ, GPTQ) | Not implemented |
| Preemption / request eviction | Not implemented |

The goal is to understand the **prefill/decode split**, **KV cache reuse**, **continuous batching**, **paged memory management**, and **prefix caching** — the core ideas everything else in vLLM is built on top of.
