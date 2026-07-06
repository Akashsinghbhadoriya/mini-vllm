from core.server import Server
from client import client
import threading

server = Server()
server.start()

thread1 = threading.Thread(
    target=client,
    args=(
        server,
        "Explain transformers."
    )
)

thread2 = threading.Thread(
    target=client,
    args=(
        server,
        "Explain LoRA."
    )
)

thread3 = threading.Thread(
    target=client,
    args=(
        server,
        "Explain FlashAttention."
    )
)

thread1.start()
thread2.start()
thread3.start()

thread1.join()
thread2.join()
thread3.join()

server.engine.stop()