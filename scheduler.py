from request import RequestStatus

class Scheduler:

    def __init__(self):
        
        self.active_requests = []

    def add_request(self, request):

        self.active_requests.append(request)

    def get_batch(self):

        return self.active_requests
    
    def remove_finished(self):

        self.active_requests = [r for r in self.active_requests if r.status != RequestStatus.FINISHED]

    def has_pending_requests(self):

        return len(self.active_requests) > 0