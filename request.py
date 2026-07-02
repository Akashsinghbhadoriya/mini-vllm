from enum import Enum, auto
import threading

class RequestStatus(Enum):

    STARTED = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()

class Request:
    def __init__(self, requestId, prompt, max_new_tokens=50, status: RequestStatus = RequestStatus.STARTED):
        super().__init__()

        self.request_id = requestId
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.generated_token_ids = []
        self.status = status
        self.prompt_token_ids = None
        self.last_token_id = None
        self.past_key_values = None
        self.generated_text = None
        self.completed = threading.Event()
        self.start_time = None
        self.end_time = None

    def mark_running(self):
        self.status = RequestStatus.RUNNING

    def mark_finished(self):
        self.status = RequestStatus.FINISHED

    def mark_failed(self):
        self.status = RequestStatus.FAILED

    def mark_waiting(self):
        self.status = RequestStatus.WAITING