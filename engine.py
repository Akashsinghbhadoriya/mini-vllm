from request import RequestStatus
import torch

class Engine:

    def __init__(self, model_runner, scheduler):
        
        self.model_runner = model_runner
        self.scheduler = scheduler
        self.all_requests = []

        self.model_runner.load_model()

    def generate(self, request):

        return self.generate_batch([request])
    
    def generate_batch(self, requests):

        self.all_requests = requests
        self.scheduler.add_requests(requests)
        # Prefill all the Requests
        batch = self.scheduler.get_batch()
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