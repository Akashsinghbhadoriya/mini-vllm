from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
from models.llama_model import LlamaModel

model_name = "meta-llama/Llama-3.2-3B"
model = AutoModelForCausalLM.from_pretrained(model_name)
llama_model = LlamaModel(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(model)
print(model.model)
print(model.config)
input_txt = "Hi this is Akash"

inputs = tokenizer(input_txt, return_tensors="pt")

# print(inputs)
outputs = model(
    input_ids = inputs["input_ids"], 
    attention_mask = inputs["attention_mask"],
    use_cache = True
    )

# print(outputs)

output_logits = outputs.logits
past_key_values = outputs.past_key_values

logits = output_logits[:,-1,:]
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples = 1)
# print(idx_next)
next_text = tokenizer.decode(idx_next)
# print(next_text)

# print(past_key_values)

# for layer in past_key_values.layers:
    # print(layer.keys)
    # print(layer.values)

# ── Step 6: Sanity check — HF logits vs MiniLlamaModel logits ──────────────
seq_len = inputs["input_ids"].shape[1]
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

with torch.no_grad():
    mini_logits, _ = llama_model(inputs["input_ids"], position_ids)

hf_last   = output_logits[:, -1, :]   # already computed above
mini_last = mini_logits[:, -1, :]

max_diff = (hf_last - mini_last).abs().max().item()
print(f"[Step 6] Max absolute logit difference (HF vs Mini): {max_diff:.6f}")

assert max_diff < 1e-3, (
    f"Logits diverged! Max diff {max_diff:.6f} exceeds 1e-3 threshold. "
    "Do NOT proceed to Step 7."
)
if max_diff < 1e-4:
    print("[Step 6] PASS — logits match within 1e-4.")
else:
    print(f"[Step 6] WARN — logits within 1e-3 but not 1e-4 (diff={max_diff:.6f}). Investigate before Step 7.")