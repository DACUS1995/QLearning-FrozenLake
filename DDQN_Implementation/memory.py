import numpy as np
import random
from typing import List, Tuple
from collections import deque

class MemoryReplay():
    def __init__(self, size: int) -> None:
        self._size = size
        self._memory = deque(maxlen=self._size)

    def get_sample(self, batch_size=1) -> List[Tuple]:
        if len(self._memory) < batch_size:
            raise Exception("Requested batch size bigger than available samples")

        return random.sample(self._memory, batch_size)


    def add_sample(self, sample: Tuple) -> None:
        self._memory.append(sample)

    def __len__(self) -> int:
        return len(self._memory)