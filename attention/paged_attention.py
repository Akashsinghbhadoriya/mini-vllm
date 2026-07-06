import torch
import torch.nn.functional as F
from attention.attention import Attention
from attention.rope import apply_rotary_emb

class PagedAttention(Attention):

    def forward(self, hidden_states, position_ids, block_table=None, kv_seq_len=0):

        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if isinstance(block_table, tuple):
            # Tensor KV cache path (decode_batch passes (k, v) tensors)
            past_k, past_v = block_table
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        elif block_table is not None and kv_seq_len > 0:
            # Paged block table decode path
            past_k = self._gather_from_blocks(block_table, kv_seq_len, is_key=True)
            past_v = self._gather_from_blocks(block_table, kv_seq_len, is_key=False)
            self._write_to_blocks(block_table, k, v, kv_seq_len)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        elif block_table is not None:
            # Prefill with block table: write KV to blocks
            self._write_to_blocks(block_table, k, v, 0)
        # else: block_table is None → prefill without paging, no-op

        new_k, new_v = k, v  # full accumulated KV (past + new) for tensor-cache path

        if self.num_key_value_heads != self.num_attention_heads:
            repeat = self.num_attention_heads // self.num_key_value_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=(block_table is None))
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)
        return out, (new_k, new_v)
    
    def _gather_from_blocks(self, block_table, kv_seq_len, is_key):
        chunks = []
        remaining = kv_seq_len
        for block in block_table.blocks:
            if remaining <= 0:
                break
            cache = block.k_cache if is_key else block.v_cache
            n = min(block.used_tokens, remaining)
            chunks.append(cache[:, :, :n, :])
            remaining -= n
        return torch.cat(chunks, dim=2)
    
    def _write_to_blocks(self, block_table, k, v, start_pos):

        seq_len = k.shape[2]
        written = 0
        for block in block_table.blocks:
            if written >= seq_len:
                break
            space = block.remaining_capacity()
            if space == 0:
                continue
            n = min(space, seq_len - written)
            offset = block.used_tokens
            if block.k_cache is None:
                block.k_cache = torch.zeros(1, self.num_key_value_heads, block.capacity, self.head_dim)
                block.v_cache = torch.zeros_like(block.k_cache)
            block.k_cache[:, :, offset:offset+n, :] = k[:, :, written:written+n, :]
            block.v_cache[:, :, offset:offset+n, :] = v[:, :, written:written+n, :]
            written += n