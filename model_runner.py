from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F

class ModelRunner:

    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        
        self.model_name = model_name 

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.eos_token_id = self.tokenizer.eos_token_id
    
    def prefill(self, request):

        inputs = self.tokenizer(request.prompt, return_tensors="pt")

        prompt_token_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        outputs = self.model(
            input_ids = prompt_token_ids,
            attention_mask = attn_mask,
            use_cache = True
        )
        output_logits = outputs.logits
        past_key_values = outputs.past_key_values

        logits = output_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        last_token_id = torch.multinomial(probs, num_samples = 1)

        #updating the request state
        request.prompt_token_ids = prompt_token_ids
        request.generated_token_ids.append(last_token_id)
        request.last_token_id = last_token_id
        request.past_key_values = past_key_values

    def decode_one_step(self, request):

        outputs = self.model(
            input_ids = request.last_token_id,
            past_key_values = request.past_key_values,
            use_cache = True
        )

        output_logits = outputs.logits
        past_key_values = outputs.past_key_values

        logits = output_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        last_token_id = torch.multinomial(probs, num_samples = 1)

        #updating the request state
        request.generated_token_ids.append(last_token_id)
        request.last_token_id = last_token_id
        request.past_key_values = past_key_values

    def decode_tokens(self, token_ids):

        return self.tokenizer.decode(
            token_ids, 
            skip_special_tokens=True
        )





