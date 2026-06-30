from request import RequestStatus
import torch

class Engine:

    def __init__(self, model_runner):
        
        self.active_requests = []
        self.model_runner = model_runner

        self.model_runner.load_model()

    def add_request(self, request):

        self.active_requests.append(request)
    
    def run(self):

        # Prefill all the Requests
        for request in self.active_requests:
            self.model_runner.prefill(request)
            request.mark_running()
        
        while not self.all_finished():

            for request in self.active_requests:

                if request.status == RequestStatus.FINISHED:
                    continue

                self.model_runner.decode_one_step(request)

                self.check_stop_conditions(request)
        
        outputs = []

        for r in self.active_requests:
            
            if r.status == RequestStatus.FINISHED:
                token_ids = torch.cat([r.prompt_token_ids, torch.cat(r.generated_token_ids,dim=1)], dim=1)
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