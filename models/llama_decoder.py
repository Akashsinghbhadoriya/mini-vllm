import torch
import torch.nn as nn
from torch.nn import functional as F

class LlamaDecoderLayer(nn.Module):

    def __init__(self, hf_layer, attention_backend):
        super().__init__()
        self.input_layernorm = hf_layer.input_layernorm
        self.post_attention_layernorm = hf_layer.post_attention_layernorm
        self.mlp = hf_layer.mlp
        self.self_attn = attention_backend

    def forward(self, hidden_states, position_ids, kv_cache=None):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, new_kv = self.self_attn(hidden_states, position_ids, kv_cache)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv