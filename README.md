# miniVLLM

A minimal from-scratch implementation of the core ideas behind [vLLM](https://github.com/vllm-project/vllm) — built for learning and understanding how modern LLM inference engines work.

## What This Is

vLLM achieves up to 24x better throughput than naive HuggingFace inference by rethinking how GPU memory is managed during generation. This project implements the foundational concepts:

- **Request lifecycle** with explicit status transitions (STARTED → RUNNING → FINISHED)
- **Prefill phase** — processes the full prompt, builds the initial KV cache
- **Decode phase** — generates one token at a time, reusing cached KV values
- **Engine loop** — batches multiple requests and drives them through prefill + decode

## Project Structure

```
miniVllm/
├── main.py              # Entry point — creates requests and runs the engine
├── engine.py            # Core inference loop (prefill all, then decode until done)
├── model_runner.py      # Wraps HuggingFace model: prefill() and decode_one_step()
├── request.py           # Request dataclass + RequestStatus enum
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

### 2. Engine

```
prefill(req)  →  mark_running()
     ↓
decode_one_step(req)  →  check_stop_conditions()
     ↓
repeat until all requests are FINISHED
```

The engine prefills all requests first, then enters a decode loop. Each decode step feeds only the last generated token (not the full sequence) and reuses `past_key_values` — this is the KV cache at work.

### 3. ModelRunner

Wraps `AutoModelForCausalLM` with two methods:

- **`prefill(request)`** — tokenizes the prompt, runs a full forward pass, samples the first output token, stores `past_key_values` on the request
- **`decode_one_step(request)`** — runs a forward pass on just the last token with cached KV values, samples the next token

Sampling is done via softmax + multinomial sampling (temperature=1).

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
