from enum import Enum, auto
import threading
from kv_cache.block_table import BlockTable
import queue

class RequestStatus(Enum):

    STARTED = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()

class Request:
    def __init__(self, 
                 requestId, 
                 prompt, 
                 max_new_tokens=50, 
                 status: RequestStatus = RequestStatus.STARTED,
                 streaming: bool = False,
                 token_queue: queue.Queue | None = None
        ):
        super().__init__()

        self.request_id = requestId
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.generated_token_ids = []
        self.status = status
        self.prompt_token_ids = None
        self.last_token_id = None
        self.past_key_values = None
        self.kv_seq_len = 0
        self.generated_text = None
        self.completed = threading.Event()
        self.start_time = None
        self.end_time = None
        self.first_token_time = None
        self.block_table = BlockTable()
        self.kv_cache = None
        self.cached_prefix_len: int = 0
        self.streaming = streaming
        # Per-phase timing accumulators (seconds)
        self.t_tokenize = 0.0
        self.t_prefix_lookup = 0.0
        self.t_prefill = 0.0
        self.t_kv_write = 0.0
        self.t_decode_total = 0.0
        self.n_decode_steps = 0
        if streaming:
            self.token_queue = queue.Queue()

    def mark_running(self):
        self.status = RequestStatus.RUNNING

    def mark_finished(self):
        self.status = RequestStatus.FINISHED

    def mark_failed(self):
        self.status = RequestStatus.FAILED

    def mark_waiting(self):
        self.status = RequestStatus.WAITING