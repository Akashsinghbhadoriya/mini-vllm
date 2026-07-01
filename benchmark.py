from engine import Engine
from model_runner import ModelRunner
from request import Request
from scheduler import Scheduler
from datetime import datetime

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

def print_summary():

    t1 = benchmark_sequential()
    t2 = benchmark_batch()
    print("============================")
    print("Mini-vLLM Benchmark")
    print("============================")
    print("Requests:", len(prompts))
    print("Model:", MODEL_NAME)
    print("\nSequential\n")
    print(f"Time:{t1} s")
    print("\nBatch\n")
    print(f"Time:{t2} s")
    print("\nSpeedup:", t1 / t2)


def main():
    print_summary()

if __name__=="__main__":
    main()