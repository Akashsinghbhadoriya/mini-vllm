from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F

model_name = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_txt = "Hi this is Akash"

inputs = tokenizer(input_txt, return_tensors="pt")

print(inputs)
outputs = model(
    input_ids = inputs["input_ids"], 
    attention_mask = inputs["attention_mask"],
    use_cache = True
    )

print(outputs)

output_logits = outputs.logits
past_key_values = outputs.past_key_values

logits = output_logits[:,-1,:]
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples = 1)
print(idx_next)
next_text = tokenizer.decode(idx_next)
print(next_text)

print(past_key_values)