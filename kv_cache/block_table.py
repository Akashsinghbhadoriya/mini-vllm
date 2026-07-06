from kv_cache.memory_block import MemoryBlock

class BlockTable:

    def __init__(self):

        self.blocks = []

    def append(self, block: MemoryBlock):
        
        self.blocks.append(block)

    def last_block(self):
        
        if len(self.blocks) == 0:
            return None
        
        return self.blocks[-1]
    
    def num_blocks(self):

        return len(self.blocks)

    def total_capacity(self):
        total = 0

        for block in self.blocks:
            total += block.capacity
        
        return total

    def used_tokens(self):
        total = 0

        for block in self.blocks:
            total += block.used_tokens

        return total
    
    def clear(self):

        self.blocks.clear()