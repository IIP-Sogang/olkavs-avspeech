class AbstractScheduler:
    def __init__(self, lr, *args, **kwargs):
        self.lr = lr
        
    def __call__(self, *args, **kwargs):
        pass
        
    def get_lr(self):
        return [self.lr]
        
    def step(self, *args, **kwargs):
        pass