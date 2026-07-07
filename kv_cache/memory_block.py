from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MemoryBlock:
        
    block_id: int
    capacity: int
    used_tokens: int = 0
    is_allocated: bool = False
    ref_count: int = 0
    token_hash: Optional[int] = None
    layer_kv: dict = field(default_factory=dict)
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
        self.ref_count = 0
        self.token_hash = None
        self.layer_kv = {}

        self.k_cache = None
        self.v_cache = None