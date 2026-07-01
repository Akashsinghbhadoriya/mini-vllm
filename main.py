from engine import Engine
from model_runner import ModelRunner
from request import Request
from scheduler import Scheduler

req1 = Request(1, "what is your name")
req2 = Request(2, "What is the capital of India")
req3 = Request(3, "how many countries in the world?")

model_runner = ModelRunner()
scheduler = Scheduler()

engine = Engine(model_runner, scheduler)

engine.add_request(req1)
engine.add_request(req2)
engine.add_request(req3)

output = engine.run()

print(output)