from core.engine import Engine
from core.model_runner import ModelRunner
from request.request import Request
from core.scheduler import Scheduler
from datetime import datetime
import time
import threading
from client import client
from core.server import Server

prompts = [
    "What is artificial intelligence?",
    "Explain transformers in simple terms.",
    "Write a short poem about the moon.",
    "What is reinforcement learning?",
    "Explain quantum computing in one paragraph."
]
MODEL_NAME = "meta-llama/Llama-3.2-3B"

def generate_test_requests():

    requests = []

    for i, p in enumerate(prompts):
        requests.append(Request(i, p))

    return requests

def benchmark_sequential():

    start = datetime.now()
    model_runner = ModelRunner(MODEL_NAME)
    scheduler = Scheduler()
    engine = Engine(model_runner, scheduler)
    requests = generate_test_requests()
    for r in requests:
        output = engine.generate(r)
    end = datetime.now()
    return (end - start).total_seconds()

def benchmark_batch():
    start = datetime.now()
    model_runner = ModelRunner(MODEL_NAME)
    scheduler = Scheduler()
    engine = Engine(model_runner, scheduler)
    requests = generate_test_requests()
    output = engine.generate_batch(requests)
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


def main():
    print_summary()

if __name__=="__main__":
    main()