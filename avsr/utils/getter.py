from torch.optim import Adam

from .loss import Hybrid_Loss, Attention_Loss, CTC_Loss
from .metric import Metric
from .search import SearchSequence
from ..scheduler.schedulers import *

def get_criterion(
    loss_fn : str = 'hybrid',
    ignore_index : int = None,
    label_smoothing : float = 0.0,
    blank_id : int = None,
):  
    if loss_fn=='hybrid':
        criterion = Hybrid_Loss(ignore_index = ignore_index, label_smoothing = label_smoothing, blank_id = blank_id)
    elif loss_fn=='att':
        criterion = Attention_Loss(ignore_index, label_smoothing)
    elif loss_fn=='ctc':
        criterion = CTC_Loss(blank_id)
    return criterion

def get_metric(vocab, log_path, unit:str='character', error_type:str='cer'):
    if unit=='character' and error_type=='ger': return None
    return Metric(vocab, log_path, unit=unit, error_type=error_type)
    
def get_optimizer(
    params,
    learning_rate,
    scheduler : str = 'noam',
    epochs : int = None,
    warmup : int = None,
    steps_per_epoch : int = 0,
    init_lr : float = None,
    final_lr : float = None,
    gamma : float = 0.1,
):
    optimizer = Adam(params, learning_rate)
    
    scheduler = Scheduler(
        method = scheduler,
        optimizer = optimizer,
        lr = learning_rate,
        init_lr = init_lr,
        final_lr = final_lr,
        gamma = gamma,
        epochs = epochs,
        warmup = warmup,
        steps_per_epoch = steps_per_epoch,
    )
    return optimizer, scheduler
    
def select_search(
    model,
    vocab_size,
    method:str='default',
    pad_id:int=0,
    sos_id:int=1,
    eos_id:int=2,
    unk_id:int=3,
    max_len : int = 150,
    ctc_rate:float=0.3,
    mp_num:int=0,
):
    assert method in ['default','hybrid','att','ctc'], f"Search method '{method}' doesn't exists. Check 'search_method' from the option file!"
    if method=='default': method = 'hybrid'
    
    return SearchSequence(
        model = model,
        method = method,
        vocab_size=vocab_size,
        pad_id = pad_id,
        sos_id = sos_id,
        eos_id = eos_id,
        unk_id = unk_id,
        max_len = max_len,
        ctc_rate = ctc_rate,
        mp_num = mp_num,
    )