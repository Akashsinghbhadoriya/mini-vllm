import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from attention.rope import apply_rotary_emb

class Attention(nn.Module):

    def __init__ (self, hf_attn, rotary_emb, config):
        super().__init__()

        self.q_proj = hf_attn.q_proj
        self.k_proj = hf_attn.k_proj
        self.v_proj = hf_attn.v_proj
        self.o_proj = hf_attn.o_proj
        self.rotary_emb = rotary_emb

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

    def forward(self, hidden_states, position_ids, kv_cache=None):

        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_attention_heads, self.head_dim).transpose(1,2)
        k = k.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1,2)
        v = v.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim = 2)
            v = torch.cat([past_v, v], dim = 2)

        new_k, new_v = k, v  # save pre-repeat for KV cache (num_key_value_heads)

        if self.num_key_value_heads != self.num_attention_heads:
            repeat = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim = 1)
            v = v.repeat_interleave(repeat, dim = 1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(kv_cache is None))

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)

        return out, (new_k, new_v)
