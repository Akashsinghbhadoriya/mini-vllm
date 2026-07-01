from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn import functional as F
from request import RequestStatus

class ModelRunner:

    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        
        self.model_name = model_name 

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
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

        last_token_id = self.last_token_id(output_logits)

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

        last_token_id = self.last_token_id(output_logits)

        #updating the request state
        request.generated_token_ids.append(last_token_id)
        request.last_token_id = last_token_id
        request.past_key_values = past_key_values

    def decode_tokens(self, token_ids):

        return self.tokenizer.decode(
            token_ids, 
            skip_special_tokens=True
        )
    
    def prefill_batch(self, batch):

        prompts = []
        for request in batch:
            prompts.append(request.prompt)
        
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]

        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attn_mask,
            use_cache = True
        )

        batch_output_logits = outputs.logits
        batch_past_key_values = outputs.past_key_values
        batch_last_token_id = self.last_token_id(batch_output_logits)

        for i, request in enumerate(batch):

            next_token = batch_last_token_id[i].item()
            request.generated_token_ids.append(next_token)
            request.last_token_id = next_token

            valid_length = attn_mask[i].sum().item()
            request.prompt_token_ids = input_ids[i][:valid_length].tolist()

            request.past_key_values = tuple(self.extract_request_kv(batch_past_key_values, i))

    def decode_batch(self, batch):

        last_tokens = [request.last_token_id for request in batch]
        input_ids = torch.tensor(last_tokens).unsqueeze(1)

        batched_kv = []
        num_layers = len(batch[0].past_key_values)
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for request in batch:
                layer_k, layer_v = request.past_key_values[layer_idx]
                layer_keys.append(layer_k)
                layer_values.append(layer_v)
            batch_k = torch.cat(layer_keys, dim=0)
            batch_v = torch.cat(layer_values, dim=0)
            batched_kv.append((batch_k, batch_v))
        
        outputs = self.model(
            input_ids = input_ids,
            past_key_values = batched_kv,
            use_cache = True
        )

        batch_output_logits = outputs.logits
        batch_past_key_values = outputs.past_key_values
        batch_last_token_id = self.last_token_id(batch_output_logits)

        for i, request in enumerate(batch):

            next_token = batch_last_token_id[i].item()
            request.generated_token_ids.append(next_token)
            request.last_token_id = next_token

            if next_token == self.eos_token_id or len(request.generated_token_ids) >= request.max_new_tokens:
                request.status = RequestStatus.FINISHED

            request.past_key_values = tuple(self.extract_request_kv(batch_past_key_values, i))


    def extract_request_kv(self, batch_past_key_values, batch_index):
        request_kv = []
        for layer_k, layer_v in zip(batch_past_key_values.key_cache, batch_past_key_values.value_cache):
            
            request_kv.append((
                layer_k[batch_index : batch_index + 1], 
                layer_v[batch_index : batch_index + 1]
            ))
            
        return request_kv
    
    def last_token_id(self, output_logits):
        logits = output_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        last_token_id = torch.multinomial(probs, num_samples = 1)

        return last_token_id


        






