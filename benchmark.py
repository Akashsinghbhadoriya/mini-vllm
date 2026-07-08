from core.engine import Engine
from core.model_runner import ModelRunner
from request.request import Request
from core.scheduler import Scheduler
from datetime import datetime
import time
import threading
import requests
from client import client
from core.server import Server

API_BASE = "http://localhost:8000"

prompts = [
    "What is artificial intelligence?",
    "Explain transformers in simple terms.",
    "Write a short poem about the moon.",
    "What is reinforcement learning?",
    "Explain quantum computing in one paragraph."
]
MODEL_NAME = "meta-llama/Llama-3.2-3B"

def generate_test_requests():

    reqs = []

    for i, p in enumerate(prompts):
        reqs.append(Request(i, p))

    return reqs

def benchmark_sequential():

    start = datetime.now()
    model_runner = ModelRunner(MODEL_NAME)
    scheduler = Scheduler()
    engine = Engine(model_runner, scheduler)
    reqs = generate_test_requests()
    for r in reqs:
        output = engine.generate(r)
    end = datetime.now()
    return (end - start).total_seconds()

def benchmark_batch():
    start = datetime.now()
    model_runner = ModelRunner(MODEL_NAME)
    scheduler = Scheduler()
    engine = Engine(model_runner, scheduler)
    reqs = generate_test_requests()
    output = engine.generate_batch(reqs)
    end = datetime.now()
    return (end - start).total_seconds()

def continuous_batching():
    threads = []
    server = Server()
    for prompt in prompts:
        t = threading.Thread(
            target=client,
            args=(server, prompt)
        )
        threads.append(t)
    start_time = time.time()
    server.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    end_time = time.time()
    server.engine.stop()
    return end_time - start_time

def benchmark_api_sequential(base_url=API_BASE):
    results = []
    for prompt in prompts:
        payload = {"prompt": prompt, "max_tokens": 128, "stream": False}
        t0 = time.time()
        resp = requests.post(f"{base_url}/v1/completions", json=payload)
        t1 = time.time()
        data = resp.json()
        m = data.get("metrics") or {}
        results.append({
            "prompt": prompt[:40],
            "client_ms": round((t1 - t0) * 1000, 1),
            "server_latency_ms": m.get("latency_ms"),
            "ttft_ms": m.get("ttft_ms"),
            "tokens_per_sec": m.get("tokens_per_sec"),
        })
    return results


def benchmark_api_concurrent(base_url=API_BASE):
    results = []
    lock = threading.Lock()

    def send_streaming(prompt):
        payload = {"prompt": prompt, "max_tokens": 128, "stream": True}
        t0 = time.time()
        ttft_ms = None
        error = None
        try:
            with requests.post(f"{base_url}/v1/completions", json=payload, stream=True) as resp:
                for line in resp.iter_lines():
                    if line and line.startswith(b"data:") and b"[DONE]" not in line:
                        if ttft_ms is None:
                            ttft_ms = round((time.time() - t0) * 1000, 1)
        except requests.exceptions.ChunkedEncodingError as e:
            error = "ChunkedEncodingError (stream cut short)"
        except Exception as e:
            error = str(e)
        total_ms = round((time.time() - t0) * 1000, 1)
        with lock:
            results.append({"prompt": prompt[:40], "ttft_ms": ttft_ms, "total_ms": total_ms, "error": error})

    threads = [threading.Thread(target=send_streaming, args=(p,)) for p in prompts]
    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = round((time.time() - t_start) * 1000, 1)
    return results, wall_ms


def print_summary():

    t1 = benchmark_sequential()
    t2 = benchmark_batch()
    t3 = continuous_batching()
    print("============================")
    print("Mini-vLLM Benchmark")
    print("============================")
    print("Requests:", len(prompts))
    print("Model:", MODEL_NAME)
    print("\nSequential\n")
    print(f"Time:{t1} s")
    print("\nBatch\n")
    print(f"Time:{t2} s")
    print("\nContinous Batching\n")
    print(f"Time:{t3} s")
    print("\nSpeedup:", t1 / t2)

    print("\n============================")
    print("API Benchmark (requires server running at", API_BASE, ")")
    print("============================")
    try:
        seq = benchmark_api_sequential()
        print("\nSequential non-streaming:\n")
        for r in seq:
            print(f"  \"{r['prompt']}\"")
            print(f"    client={r['client_ms']}ms  server={r['server_latency_ms']}ms  ttft={r['ttft_ms']}ms  tok/s={r['tokens_per_sec']}")
        conc, wall_ms = benchmark_api_concurrent()
        print("\nConcurrent streaming (5 users):\n")
        for r in conc:
            print(f"  \"{r['prompt']}\"")
            suffix = f"  [{r['error']}]" if r.get("error") else ""
            print(f"    ttft={r['ttft_ms']}ms  total={r['total_ms']}ms{suffix}")
        print(f"\n  Wall time (all {len(prompts)} concurrent): {wall_ms}ms")
    except requests.exceptions.ConnectionError:
        print("  ERROR: Could not connect to API server. Start it with:")
        print("  uvicorn api.app:app --host 0.0.0.0 --port 8000")


def main():
    print_summary()

if __name__=="__main__":
    main()