from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
from torch.nn import functional as F
from request.request import RequestStatus
from models.llama_model import LlamaModel

class ModelRunner:

    def __init__(self, model_name="meta-llama/Llama-3.2-3B"):
        
        self.model_name = model_name 

    def load_model(self):
        self.device = torch.device("cpu")
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.model = LlamaModel(hf_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model loaded on {self.device} (bfloat16)")

    def tokenize_batch(self, batch):
        """Tokenize prompts and store plain token-id lists on each request."""
        prompts = [r.prompt for r in batch]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        for i, request in enumerate(batch):
            valid_length = int(attn_mask[i].sum().item())
            request.prompt_token_ids = input_ids[i][:valid_length].tolist()


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
    
    @torch.inference_mode()
    def prefill_batch(self, batch):
        no_cache = [r for r in batch if r.cached_prefix_len == 0]
        has_cache = [r for r in batch if r.cached_prefix_len > 0]

        if no_cache:
            prompts = [r.prompt for r in no_cache]
            inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attn_mask = inputs["attention_mask"].to(self.device)
            position_ids = attn_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attn_mask == 0, 0)
            logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
            batch_last_token = self.sample(logits)

            for i, request in enumerate(no_cache):
                valid_length = int(attn_mask[i].sum().item())
                request.prompt_token_ids = input_ids[i][:valid_length].tolist()
                request.last_token_id = batch_last_token[i].item()
                request.generated_token_ids.append(request.last_token_id)
                request.kv_cache = [
                    (k[i:i+1, :, -valid_length:, :],
                     v[i:i+1, :, -valid_length:, :])
                    for k, v in kv_caches
                ]
                request.kv_seq_len = valid_length

        # Separate fully-cached requests (edge case) from partial-cache hits
        fully_cached = [r for r in has_cache
                        if len(r.prompt_token_ids[r.cached_prefix_len:]) == 0]
        partial_cached = [r for r in has_cache
                          if len(r.prompt_token_ids[r.cached_prefix_len:]) > 0]

        for request in fully_cached:
            self._prefill_with_prefix_cache(request)

        # Group partial-cache requests by (cached block IDs, suffix length)
        # so each group can be prefilled in a single batched forward pass
        groups: dict = {}
        for r in partial_cached:
            suffix_len = len(r.prompt_token_ids) - r.cached_prefix_len
            key = (tuple(b.block_id for b in r.block_table.blocks), suffix_len)
            groups.setdefault(key, []).append(r)

        for group in groups.values():
            self._prefill_batch_with_prefix_cache(group)

    @torch.inference_mode()
    def _prefill_with_prefix_cache(self, request):
        """Run the model only on suffix tokens, reusing cached prefix KV."""
        cached_blocks = request.block_table.blocks
        num_layers = len(cached_blocks[0].layer_kv)
        cached_prefix_len = request.cached_prefix_len

        # Reconstruct per-layer cached KV by concatenating blocks
        cached_kv = []
        for layer_idx in range(num_layers):
            k_chunks = [b.layer_kv[layer_idx][0] for b in cached_blocks]
            v_chunks = [b.layer_kv[layer_idx][1] for b in cached_blocks]
            cached_kv.append((
                torch.cat(k_chunks, dim=2),
                torch.cat(v_chunks, dim=2)
            ))

        suffix_ids = request.prompt_token_ids[cached_prefix_len:]

        # Edge case: whole prompt is cached — peel last token to get fresh logits
        if len(suffix_ids) == 0:
            suffix_ids = [request.prompt_token_ids[-1]]
            cached_kv = [(k[:, :, :-1, :], v[:, :, :-1, :]) for k, v in cached_kv]
            cached_prefix_len -= 1

        input_ids = torch.tensor([suffix_ids], device=self.device)
        position_ids = torch.arange(
            cached_prefix_len,
            cached_prefix_len + len(suffix_ids),
            device=self.device
        ).unsqueeze(0)

        # PagedAttention sees cached_kv as tuple → tensor KV path; cats internally
        logits, new_kv_caches = self.model(input_ids, position_ids, kv_caches=cached_kv)

        # new_kv_caches already holds full accumulated KV (cached prefix + suffix)
        request.kv_cache = [(k[0:1], v[0:1]) for k, v in new_kv_caches]
        request.kv_seq_len = cached_prefix_len + len(suffix_ids)
        request.last_token_id = self.sample(logits)[0].item()
        request.generated_token_ids.append(request.last_token_id)

    @torch.inference_mode()
    def _prefill_batch_with_prefix_cache(self, requests):
        """Batch-prefill requests that share the same cached prefix blocks and suffix length."""
        rep = requests[0]
        cached_blocks = rep.block_table.blocks
        num_layers = len(cached_blocks[0].layer_kv)
        cached_prefix_len = rep.cached_prefix_len

        # Build cached_kv once — all requests in this group share identical cached blocks
        cached_kv = []
        for layer_idx in range(num_layers):
            k_chunks = [b.layer_kv[layer_idx][0] for b in cached_blocks]
            v_chunks = [b.layer_kv[layer_idx][1] for b in cached_blocks]
            cached_kv.append((
                torch.cat(k_chunks, dim=2),
                torch.cat(v_chunks, dim=2),
            ))

        suffix_lists = [r.prompt_token_ids[r.cached_prefix_len:] for r in requests]
        suffix_len = len(suffix_lists[0])  # same for all by construction
        B = len(requests)

        input_ids = torch.tensor(suffix_lists, device=self.device)  # [B, suffix_len]
        position_ids = torch.arange(
            cached_prefix_len, cached_prefix_len + suffix_len, device=self.device
        ).unsqueeze(0).expand(B, -1)  # [B, suffix_len]

        # Expand cached_kv from [1, heads, cached_len, head_dim] to [B, ...]
        batched_cached_kv = [
            (k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1))
            for k, v in cached_kv
        ]

        logits, new_kv_caches = self.model(input_ids, position_ids, kv_caches=batched_cached_kv)
        sampled = self.sample(logits)  # [B]

        for i, request in enumerate(requests):
            request.last_token_id = sampled[i].item()
            request.generated_token_ids.append(request.last_token_id)
            request.kv_cache = [
                (k[i:i+1], v[i:i+1])
                for k, v in new_kv_caches
            ]
            request.kv_seq_len = cached_prefix_len + suffix_len

    @torch.inference_mode()
    def decode_batch(self, batch):
        B = len(batch)
        last_tokens = [request.last_token_id for request in batch]
        input_ids = torch.tensor(last_tokens, device=self.device).unsqueeze(1)  # [B] -> [B, 1]

        kv_seq_lens = [request.kv_seq_len for request in batch]
        max_kv_len = max(kv_seq_lens)
        position_ids = torch.tensor([[kv_seq_lens[i]] for i in range(B)], device=self.device)

        batched_kv = []
        num_layers = len(batch[0].kv_cache)
        
        for layer_idx in range(num_layers):
            layer_keys, layer_vals = [], []
            for i, request in enumerate(batch):
                k, v = request.kv_cache[layer_idx]
                pad = max_kv_len - kv_seq_lens[i]
                if pad > 0:
                    k = F.pad(k, (0, 0, pad, 0))
                    v = F.pad(v, (0, 0, pad, 0))
                layer_keys.append(k)
                layer_vals.append(v)
            batched_kv.append((
                torch.cat(layer_keys, dim=0),
                torch.cat(layer_vals, dim=0)
            ))
        
        logits, new_kv_caches = self.model(input_ids, position_ids, kv_caches = batched_kv)
        batch_last_token = self.sample(logits)

        for i, request in enumerate(batch):
            next_token = batch_last_token[i].item()
            request.generated_token_ids.append(next_token)
            request.last_token_id = next_token

            if next_token == self.eos_token_id or len(request.generated_token_ids) >= request.max_new_tokens:
                request.status = RequestStatus.FINISHED

            request.kv_cache = [
                (k[i:i+1, :, max_kv_len - kv_seq_lens[i]:, :],
                 v[i:i+1, :, max_kv_len - kv_seq_lens[i]:, :])
                 for k, v in new_kv_caches
            ]
            request.kv_seq_len += 1
        # for layer_idx in range(num_layers):
        #     layer_keys = []
        #     layer_values = []
        #     for request, kv_seq_len in zip(batch, kv_seq_lens):
        #         layer_k, layer_v = request.past_key_values[layer_idx]
        #         # Strip left-padding accumulated from previous decode steps
        #         tensor_len = layer_k.shape[2]
        #         trim_start = tensor_len - kv_seq_len
        #         layer_k = layer_k[:, :, trim_start:, :]
        #         layer_v = layer_v[:, :, trim_start:, :]
        #         # Left-pad shorter caches to max_kv_len
        #         pad_len = max_kv_len - kv_seq_len
        #         if pad_len > 0:
        #             layer_k = F.pad(layer_k, (0, 0, pad_len, 0))
        #             layer_v = F.pad(layer_v, (0, 0, pad_len, 0))
        #         layer_keys.append(layer_k)
        #         layer_values.append(layer_v)
        #     batch_k = torch.cat(layer_keys, dim=0)
        #     batch_v = torch.cat(layer_values, dim=0)
        #     batched_kv.update(batch_k, batch_v, layer_idx)

        # # Attention mask: 0 for left-padding, 1 for real KV positions + new token
        # attention_mask = torch.zeros(len(batch), max_kv_len + 1)
        # for i, kv_seq_len in enumerate(kv_seq_lens):
        #     attention_mask[i, max_kv_len - kv_seq_len:] = 1

        # outputs = self.model(
        #     input_ids=input_ids,
        #     past_key_values=batched_kv,
        #     attention_mask=attention_mask,
        #     use_cache=True
        # )

        # batch_output_logits = outputs.logits
        # batch_past_key_values = outputs.past_key_values
        # batch_last_token_id = self.last_token_id(batch_output_logits)

        # for i, request in enumerate(batch):

        #     next_token = batch_last_token_id[i].item()
        #     request.generated_token_ids.append(next_token)
        #     request.last_token_id = next_token

        #     if next_token == self.eos_token_id or len(request.generated_token_ids) >= request.max_new_tokens:
        #         request.status = RequestStatus.FINISHED

        #     request.past_key_values = tuple(self.extract_request_kv(batch_past_key_values, i))
        #     request.kv_seq_len += 1


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
    
    def sample(self, logits):
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)
    
    def decode_single_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)


        






