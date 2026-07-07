from collections import deque
from kv_cache.memory_block import MemoryBlock
from kv_cache.prefix_cache import PrefixCache

class KVCacheManager:

    def __init__(self, num_blocks = 1024, block_size = 16, max_cache_blocks=256):
        
        self.block_size = block_size
        self.blocks = []
        self.free_blocks = deque()
        self.allocated_blocks = {}
        self.prefix_cache = PrefixCache(max_cache_blocks)

        for i in range(num_blocks):

            block = MemoryBlock(
                block_id=i,
                capacity=block_size
            )

            self.blocks.append(block)
            self.free_blocks.append(block)

    def _hash(self, token_ids_tuple: int):
        return hash(token_ids_tuple)
    
    def lookup_prefix(self, token_ids: list):
        matched = []
        for i in range(0,len(token_ids),self.block_size):
            end = min(i + self.block_size, len(token_ids))
            key = tuple(token_ids[i:end])
            block = self.prefix_cache.lookup(hash(key))
            if block is None:
                break
            block.ref_count +=1
            matched.append(block)
        return matched
    
    def write_kv_to_blocks(self, block_table, kv_caches):
        for layer_idx, (k, v) in enumerate(kv_caches):
            token_offset = 0
            for block in block_table.blocks:
                n = block.used_tokens
                block.layer_kv[layer_idx] = (
                    k[:, :, token_offset : token_offset + n, :],
                    v[:, :, token_offset : token_offset + n, :]
                )
                token_offset += n
    
    def cache_completed_blocks(self, block_table, token_ids: list):
        for i, block in enumerate(block_table.blocks):
            if block.used_tokens < block.capacity:
                break
            key = tuple(token_ids[:(i+1)*self.block_size])
            h = hash(key)
            if h in self.prefix_cache:
                continue
            block.token_hash = h
            block.ref_count += 1
            evicted = self.prefix_cache.insert(h, block)
            if evicted:
                self._decr_ref(evicted)
    
    def _decr_ref(self, block):
        block.ref_count -= 1
        if block.ref_count == 0 and self.prefix_cache.lookup(block.token_hash) is None:
            self.free_block(block)

    def allocate_block(self):

        if len(self.free_blocks) == 0:
            raise RuntimeError("Out of KV Cache Blocks")
        
        block = self.free_blocks.popleft()
        block.is_allocated = True
        self.allocated_blocks[block.block_id] = block

        return block
    
    def free_block(self, block):

        block.clear()
        self.allocated_blocks.pop(block.block_id)
        self.free_blocks.append(block)

    def allocate_for_request(self, block_table, num_tokens):
        already = block_table.used_tokens()
        remaining = num_tokens - already
        while remaining > 0:

            if block_table.total_capacity() > block_table.used_tokens():
                block = block_table.last_block
            else:
                block = self.allocate_block()

            tokens = min(remaining, self.block_size)

            block.add_tokens(tokens)
            block_table.append(block)
            remaining -= tokens

    def free_request(self, block_table):

        for block in block_table.blocks:
            self._decr_ref(block)

        block_table.clear()

    def stats(self):

        return {
            "total blocks":len(self.blocks),
            "allocated_blocks": len(self.allocated_blocks),
            "free blocks": len(self.free_blocks)
        }

