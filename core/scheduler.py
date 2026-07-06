from request.request import RequestStatus

class Scheduler:

    def __init__(self, max_batch = 8):
        
        self.active_requests = []
        self.max_batch = max_batch

    def add_active(self, requests):

        self.active_requests.extend(requests)

    def get_active(self):

        return self.active_requests
    
    def remove_finished(self):

        finished_request = [r for r in self.active_requests if r.status == RequestStatus.FINISHED]
        self.active_requests = [r for r in self.active_requests if r.status != RequestStatus.FINISHED]
        return finished_request

    def has_pending_requests(self):

        return len(self.active_requests) > 0
    
    def has_capacity(self):

        return len(self.active_requests) < self.max_batch
    
    def get_capacity(self):

        return self.max_batch - len(self.active_requests)
    
    def has_active(self):

        return len(self.active_requests) > 0