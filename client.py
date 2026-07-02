import threading

print_lock = threading.Lock()

def client(server, prompt):

    response, request_id, latency = server.submit_request(prompt)

    with print_lock:
        print("--------------------------------")
        print(f"[Request {request_id}] Latency: {latency:.3f}s")
        print("Prompt:")

        print(prompt)

        print()

        print("Response:")

        print(response)

        print("--------------------------------")