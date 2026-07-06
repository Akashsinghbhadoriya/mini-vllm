from dataclasses import dataclass

@dataclass
class MemoryBlock:
        
    block_id: int
    capacity: int
    used_tokens: int = 0
    is_allocated: bool = False

    k_cache = None
    v_cache = None

    def has_space(self):
        
        return self.used_tokens < self.capacity

    def remaining_capacity(self):
        
        return self.capacity - self.used_tokens

    def add_tokens(self, num_tokens: int):
        self.used_tokens += num_tokens

    def clear(self):

        self.used_tokens = 0
        self.is_allocated = False

        self.k_cache = None
        self.v_cache = None