from engine import Engine
from model_runner import ModelRunner
from request import Request
from scheduler import Scheduler
from request_queue import RequestQueue
from response_queue import ReponseQueue
import threading
import time
from kv_cache_manager import KVCacheManager

class Server:

    def __init__(self):
        self.request_queue = RequestQueue()
        self.response_queue = ReponseQueue()
        self.model_runner = ModelRunner()
        self.scheduler = Scheduler()
        self.kv_manager = KVCacheManager(
            num_blocks=1024,
            block_size=16
        )
        self.engine = Engine(
            self.model_runner,
            self.scheduler,
            self.request_queue,
            self.response_queue,
            self.kv_manager
        )
        self.request_counter = 0
        self.counter_lock = threading.Lock()

    def start(self):
        print("starting server")
        self.engine.start()
        print("server ready")

    def submit_request(self, prompt):

        with self.counter_lock:
            self.request_counter += 1
            request_id = self.request_counter

        request = Request(request_id, prompt)
        request.start_time = time.time()
        self.request_queue.enqueue(request)

        print(f"Submitted Request {request_id}")
        request.completed.wait()
        request.end_time = time.time()

        latency = request.end_time - request.start_time
        return request.generated_text, request_id, latency