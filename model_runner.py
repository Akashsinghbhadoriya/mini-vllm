from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
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

            valid_length = int(attn_mask[i].sum().item())
            request.prompt_token_ids = input_ids[i][:valid_length].tolist()

            request.past_key_values = tuple(
                (k[:, :, :valid_length, :], v[:, :, :valid_length, :])
                for k, v in self.extract_request_kv(batch_past_key_values, i)
            )
            request.kv_seq_len = valid_length

    def decode_batch(self, batch):

        last_tokens = [request.last_token_id for request in batch]
        input_ids = torch.tensor(last_tokens).unsqueeze(1)

        kv_seq_lens = [request.kv_seq_len for request in batch]
        max_kv_len = max(kv_seq_lens)

        batched_kv = DynamicCache()
        num_layers = len(batch[0].past_key_values)
        for layer_idx in range(num_layers):
            layer_keys = []
            layer_values = []
            for request, kv_seq_len in zip(batch, kv_seq_lens):
                layer_k, layer_v = request.past_key_values[layer_idx]
                # Strip left-padding accumulated from previous decode steps
                tensor_len = layer_k.shape[2]
                trim_start = tensor_len - kv_seq_len
                layer_k = layer_k[:, :, trim_start:, :]
                layer_v = layer_v[:, :, trim_start:, :]
                # Left-pad shorter caches to max_kv_len
                pad_len = max_kv_len - kv_seq_len
                if pad_len > 0:
                    layer_k = F.pad(layer_k, (0, 0, pad_len, 0))
                    layer_v = F.pad(layer_v, (0, 0, pad_len, 0))
                layer_keys.append(layer_k)
                layer_values.append(layer_v)
            batch_k = torch.cat(layer_keys, dim=0)
            batch_v = torch.cat(layer_values, dim=0)
            batched_kv.update(batch_k, batch_v, layer_idx)

        # Attention mask: 0 for left-padding, 1 for real KV positions + new token
        attention_mask = torch.zeros(len(batch), max_kv_len + 1)
        for i, kv_seq_len in enumerate(kv_seq_lens):
            attention_mask[i, max_kv_len - kv_seq_len:] = 1

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=batched_kv,
            attention_mask=attention_mask,
            use_cache=True
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
            request.kv_seq_len += 1


    def extract_request_kv(self, batch_past_key_values, batch_index):
        request_kv = []
        for layer in batch_past_key_values.layers:
            layer_k = layer.keys
            layer_v = layer.values
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


        






