from collections import OrderedDict
from kv_cache.memory_block import MemoryBlock

class PrefixCache:

    def __init__(self, max_blocks: int):
        self.max_blocks = max_blocks
        self._cache: OrderedDict = OrderedDict()

    def lookup(self, block_hash: int):
        if block_hash in self._cache:
            self._cache.move_to_end(block_hash)
            return self._cache[block_hash]
        return None
    
    def insert(self, block_hash: int, block) -> "MemoryBlock | None":
        if block_hash in self._cache:
            self._cache.move_to_end(block_hash)
            return None
        evicted = None
        if len(self._cache) >= self.max_blocks:
            _, evicted = self._cache.popitem(last=False)
        self._cache[block_hash] = block
        return evicted
    
    def remove(self, block_hash: int):
        self._cache.pop(block_hash, None)

    def __len__(self):
        return len(self._cache)
    
    def __contains__(self, block_hash):
        return block_hash in self._cache
