from request.request import Request
import time
import asyncio

STREAM_DONE = object()

class RequestHandle:
    def __init__(self, request: Request):
        self._request = request

    def wait(self):
        self._request.completed.wait()
        latency = time.time() - self._request.start_time
        
        return self._request.generated_text, self._request.request_id, latency
    
    async def stream(self):
        loop = asyncio.get_event_loop()
        while True:
            token = await loop.run_in_executor(None, self._request.token_queue.get)
            if token is STREAM_DONE:
                break
            yield token