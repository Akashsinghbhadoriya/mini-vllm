import torch
import torch.nn as nn
from models.llama_decoder import LlamaDecoderLayer
from attention.attention import Attention
from attention.paged_attention import PagedAttention

class LlamaModel(nn.Module):

    def __init__(self, hf_model):
        super().__init__()
        hf = hf_model.model
        self.embed_tokens = hf.embed_tokens
        self.norm = hf.norm
        self.lm_head = hf_model.lm_head
        rotary_emb = hf.rotary_emb
        config = hf_model.config

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(layer, PagedAttention(layer.self_attn, rotary_emb, config))
            for layer in hf.layers
        ])

    def forward(self, input_ids, position_ids, kv_caches=None):

        hidden_states = self.embed_tokens(input_ids)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(hidden_states, position_ids, kv)
            new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_kv_caches