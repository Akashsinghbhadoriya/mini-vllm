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
- **Server / Client** — thread-safe server that accepts concurrent client requests and tracks per-request latency

## Project Structure

```
miniVllm/
├── main.py              # Entry point — spawns concurrent client threads against the Server
├── server.py            # Server: wraps Engine with a thread-safe submit interface + latency tracking
├── client.py            # Client: submits a prompt to the server and prints the response + latency
├── engine.py            # Core inference engine (static batch mode + continuous batching serve loop)
├── model_runner.py      # Wraps HuggingFace model: prefill_batch() and decode_batch()
├── request.py           # Request dataclass + RequestStatus enum
├── scheduler.py         # Manages active request batch, capacity, and status transitions
├── request_queue.py     # Thread-safe inbound request queue (deque + lock)
├── response_queue.py    # Thread-safe outbound response queue (deque + lock)
├── benchmark.py         # Benchmarks sequential vs batch vs continuous batching throughput
├── inspect_forward.py   # Scratch file for exploring model forward pass outputs
├── vllm.md              # Notes on vLLM concepts: prefill vs decode
├── PagedAttention.md    # Notes on PagedAttention memory management
└── requirements.txt     # torch, transformers
```

## How It Works

### 1. Request

Each `Request` holds:
- The prompt string
- Token IDs generated so far
- The last generated token (used as input for next decode step)
- Cached `past_key_values` from the model (the KV cache)
- A `RequestStatus` tracking where it is in the pipeline
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

All requests are prefilled together in a single batched forward pass. The decode loop then processes all active requests jointly each step, reusing each request's `past_key_values`.

**Continuous Batching Mode** (`serve`, used by Server):

```
Engine runs in a background thread (start() / stop())
     ↓
loop:
    if scheduler has capacity:
        dequeue up to N new requests from request_queue
        prefill_batch(new_requests)         ← adds to active batch mid-flight
    if scheduler has active requests:
        decode_batch(active_batch)
        finished = scheduler.remove_finished()
        for each finished:
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

Wraps `AutoModelForCausalLM` with two batch-oriented methods:

- **`prefill_batch(requests)`** — tokenizes and pads the prompt batch, runs one joint forward pass, extracts per-request KV cache from the batched output, samples the first token for each request
- **`decode_batch(requests)`** — concatenates per-request KV caches into a batched structure, runs a forward pass on only the last tokens (one per request), samples the next token, updates each request's KV cache for the next step

Sampling uses softmax + multinomial sampling (temperature=1).

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

> Requires access to `meta-llama/Llama-3.2-3B` on HuggingFace. Set `HUGGING_FACE_HUB_TOKEN` or run `huggingface-cli login` first.

`main.py` starts a Server and submits three prompts concurrently from separate client threads. You can add more threads or change the prompts there.

You can swap the model by changing `model_name` in `model_runner.py`:

```python
model_runner = ModelRunner(model_name="gpt2")  # lighter model for testing
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

See [`vllm.md`](./vllm.md) for notes on the prefill vs decode distinction, and [`PagedAttention.md`](./PagedAttention.md) for how vLLM extends this with paged memory management.

## What's Missing (vs Real vLLM)

This is intentionally minimal. Real vLLM adds:

| Feature | Status |
|---|---|
| PagedAttention (non-contiguous KV blocks) | Not implemented |
| Continuous batching | Implemented (basic) |
| Async engine / API server | Partially implemented (threaded Server) |
| GPU memory pre-allocation | Not implemented |
| Beam search / parallel sampling | Not implemented |
| Quantization (AWQ, GPTQ) | Not implemented |
| Preemption / request eviction | Not implemented |

The goal here is to understand the **prefill/decode split**, **KV cache reuse**, and **continuous batching** — the core ideas everything else in vLLM is built on top of.
