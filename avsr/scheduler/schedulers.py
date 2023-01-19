from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from avsr.scheduler.noam import NoamLR
from avsr.scheduler.abstract import AbstractScheduler

class Scheduler:
    def __init__(
        self, 
        method = 'none',
        optimizer = None,
        lr = None ,  
        init_lr = None,
        final_lr = None,
        epochs = None,
        steps_per_epoch = None,
        warmup = None,
        gamma = None,
        *args, 
        **kwargs,
    ):
        self.method = method
        if method == 'noam':
            self.scheduler = NoamLR(
                optimizer,
                [warmup],
                [epochs],
                [steps_per_epoch], # dataset_size / batch_size (= len(dataloader))
                [init_lr],
                [lr],
                [final_lr],
            )
        elif method == 'reduce':
            self.scheduler = ExponentialLR(
                optimizer,
                gamma = gamma,
            )
        else:
            self.scheduler = AbstractScheduler(lr = lr)
        
    def get_lr(self):
        return self.scheduler.get_lr()
        
    def step(self, on='step', loss=None, step=None, *args, **kwargs):
        if on=='step':
            if self.method == 'noam':
                self.scheduler.step(step)
        elif on=='epoch':
            if self.method == 'reduce':
                self.scheduler.step()