# miniVLLM

A minimal from-scratch implementation of the core ideas behind [vLLM](https://github.com/vllm-project/vllm) — built for learning and understanding how modern LLM inference engines work.

## What This Is

vLLM achieves up to 24x better throughput than naive HuggingFace inference by rethinking how GPU memory is managed during generation. This project implements the foundational concepts:

- **Request lifecycle** with explicit status transitions (STARTED → RUNNING → FINISHED)
- **Prefill phase** — processes the full prompt batch together, builds the initial KV cache per request
- **Decode phase** — generates one token at a time across all active requests, reusing cached KV values
- **Scheduler** — manages the active request batch and drives status transitions
- **Engine loop** — orchestrates prefill + decode across a batch of requests until all finish

## Project Structure

```
miniVllm/
├── main.py              # Entry point — creates requests and runs the engine
├── engine.py            # Core inference loop (batch prefill, then batch decode until done)
├── model_runner.py      # Wraps HuggingFace model: prefill_batch() and decode_batch()
├── request.py           # Request dataclass + RequestStatus enum
├── scheduler.py         # Manages active request batches and status transitions
├── benchmark.py         # Benchmarks sequential vs batch inference throughput
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

### 2. Scheduler

The `Scheduler` manages the active request batch:
- `add_requests(requests)` — registers new requests (status: WAITING)
- `get_batch()` — returns all currently active requests
- `remove_finished()` — evicts requests that have reached FINISHED status

### 3. Engine

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

All requests are prefilled together in a single batched forward pass. The decode loop then processes all active requests jointly each step, reusing each request's `past_key_values` — this is the KV cache at work.

### 4. ModelRunner

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

You can swap the model by changing `model_name` in `model_runner.py`:

```python
model_runner = ModelRunner(model_name="gpt2")  # lighter model for testing
```

## Benchmarking

`benchmark.py` measures the throughput difference between processing requests one-by-one vs. batching them together:

```bash
python benchmark.py
```

It runs 5 requests in two modes and reports wall-clock time and the speedup ratio:

| Mode | Description |
|---|---|
| Sequential | Each request prefilled and decoded independently |
| Batch | All requests prefilled together, decoded together each step |

Batching wins by amortizing the high compute cost of prefill across multiple sequences and making better use of GPU memory bandwidth during decode.

## Key Concepts

See [`vllm.md`](./vllm.md) for notes on the prefill vs decode distinction, and [`PagedAttention.md`](./PagedAttention.md) for how vLLM extends this with paged memory management.

## What's Missing (vs Real vLLM)

This is intentionally minimal. Real vLLM adds:

| Feature | Status |
|---|---|
| PagedAttention (non-contiguous KV blocks) | Not implemented |
| Continuous batching | Not implemented |
| Async engine / API server | Not implemented |
| GPU memory pre-allocation | Not implemented |
| Beam search / parallel sampling | Not implemented |
| Quantization (AWQ, GPTQ) | Not implemented |

The goal here is to understand the **prefill/decode split** and **KV cache reuse** — the two ideas everything else in vLLM is built on top of.
