from collections import deque
import threading

class ReponseQueue:

    def __init__(self):
        
        self._queue = deque()
        self._lock = threading.Lock()

    def enqueue(self,request):

        with self._lock:
            self._queue.append(request)

    def dequeue(self):

        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()
        
    def is_empty(self):

        with self._lock:
            return len(self._queue) == 0