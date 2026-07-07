from request.request import RequestStatus
import torch
import threading

class Engine:

    def __init__(
            self, 
            model_runner, 
            scheduler, 
            request_queue = None, 
            response_queue = None,
            kv_manager = None
    ):
        
        self.model_runner = model_runner
        self.scheduler = scheduler
        self.all_requests = []
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.model_runner.load_model()
        self.kv_manager = kv_manager

    def start(self):
        self.running = True

        self.engine_thread = threading.Thread(
            target=self.serve,
            daemon=True
        )

        self.engine_thread.start()
        print("Engine started")
    
    def stop(self):

        self.running = False
        self.engine_thread.join()
        print("Engine Stopped")

    # Sequential Processing of requests
    def generate(self, request):

        return self.generate_batch([request])
    
    #static batching for benchmarking
    def generate_batch(self, requests):

        self.all_requests = requests
        self.scheduler.add_active(requests)
        # Prefill all the Requests
        batch = self.scheduler.get_active()
        self.model_runner.prefill_batch(batch)
        
        while self.scheduler.has_pending_requests():

            self.model_runner.decode_batch(batch)

            self.scheduler.remove_finished()
        
        outputs = []

        for r in self.all_requests:
            
            if r.status == RequestStatus.FINISHED:
                token_ids = r.prompt_token_ids + r.generated_token_ids
                text = self.model_runner.decode_tokens(token_ids)
                outputs.append({
                    "request_id": r.request_id,
                    "response": text
                })
        
        return outputs

    def serve(self):
        
        while self.running:

            if self.scheduler.has_capacity():
                capacity = self.scheduler.get_capacity()
                new_requests = self.request_queue.dequeue_many(capacity)
                if new_requests:
                    self.scheduler.add_active(new_requests)

                    # Tokenize first so prefix lookup has token ids
                    self.model_runner.tokenize_batch(new_requests)

                    # Prefix cache lookup: pre-populate block_table with cached blocks
                    for request in new_requests:
                        cached_blocks = self.kv_manager.lookup_prefix(request.prompt_token_ids)
                        request.cached_prefix_len = len(cached_blocks) * self.kv_manager.block_size
                        for block in cached_blocks:
                            request.block_table.append(block)

                    # Prefill: skips cached prefix for cache-hit requests
                    self.model_runner.prefill_batch(new_requests)

                    # Allocate new blocks for the suffix (cached blocks already in block_table)
                    for request in new_requests:
                        self.kv_manager.allocate_for_request(
                            request.block_table,
                            request.kv_seq_len
                        )

                    # Write KV tensors into blocks, then cache completed full blocks
                    for request in new_requests:
                        self.kv_manager.write_kv_to_blocks(request.block_table, request.kv_cache)
                        self.kv_manager.cache_completed_blocks(
                            request.block_table, request.prompt_token_ids
                        )

            if self.scheduler.has_active():
                batch = self.scheduler.get_active()

                self.model_runner.decode_batch(batch)
                for request in batch:
                    current_len = request.block_table.used_tokens() - request.kv_seq_len
                    self.kv_manager.allocate_for_request(
                        request.block_table,
                        current_len
                    )
                finished_request = self.scheduler.remove_finished()

                for request in finished_request:
                    self.decode_text(request)
                    self.response_queue.enqueue(request)
                    self.kv_manager.free_request(request.block_table)
    
    def check_stop_conditions(self, request):

        if len(request.generated_token_ids) >= request.max_new_tokens:
            request.mark_finished()

        elif request.last_token_id == self.model_runner.eos_token_id:
            request.mark_finished()

    def all_finished(self):

        for r in self.active_requests:
            if r.status != RequestStatus.FINISHED:
                return False
        return True
    
    def decode_text(self, request):
        token_ids = request.prompt_token_ids + request.generated_token_ids
        text = self.model_runner.decode_tokens(token_ids)
        request.generated_text = text
        request.completed.set()