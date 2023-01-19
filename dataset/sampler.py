from typing import Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

class DistributedCurriculumSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas,
                         rank, shuffle, seed, drop_last)
        
    def set_epoch(self, epoch:int) -> None:
        self.epoch = epoch
        if epoch == 0:
            self.shuffle = False
        else:
            self.shuffle = True
    
    