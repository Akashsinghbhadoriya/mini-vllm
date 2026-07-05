from collections import deque
from memory_block import MemoryBlock

class KVCacheManager:

    def __init__(self, num_blocks = 1024, block_size = 16):
        
        self.block_size = block_size
        self.blocks = []
        self.free_blocks = deque()
        self.allocated_blocks = {}

        for i in range(num_blocks):

            block = MemoryBlock(
                block_id=i,
                capacity=block_size
            )

            self.blocks.append(block)
            self.free_blocks.append(block)

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

        while num_tokens > 0:

            if block_table.total_capacity() > block_table.used_tokens():
                block = block_table.last_block
            else:
                block = self.allocate_block()

            tokens = min(num_tokens, self.block_size)

            block.add_tokens(tokens)
            block_table.append(block)
            num_tokens -= tokens

    def free_request(self, block_table):

        for block in block_table.blocks:
            self.free_block(block)

        block_table.clear()

    def stats(self):

        return {
            "total blocks":len(self.blocks),
            "allocated_blocks": len(self.allocated_blocks),
            "free blocks": len(self.free_blocks)
        }

