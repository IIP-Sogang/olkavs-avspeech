import torch
import torch.nn as nn


class Hybrid_Loss(nn.Module):
    '''
    Inputs : 
        outputs : tuple ( tensor (BxSxE), tensor (BxLxE) )
        targets : tensor (BxS)
    '''
    def __init__(
        self, 
        blank_id : str = None,
        ignore_index : int = None,
        ctc_rate : float = 0.2,
        label_smoothing : float = 0.0
    ):
        super().__init__()
        self.ctc_rate = ctc_rate
        self.ctc   = CTC_Loss(blank_id = blank_id)
        self.att   = Attention_Loss(ignore_index=ignore_index, label_smoothing = label_smoothing)
        
    def forward(self, outputs, targets, target_lengths, att=None, *args, **kwargs):
        att_out = outputs[0].contiguous()
        ctc_out = outputs[1].contiguous()
        att_loss = self.att(att_out, targets, target_lengths)
        ctc_loss = self.ctc(ctc_out, targets, target_lengths)
        loss = self.ctc_rate * ctc_loss + (1-self.ctc_rate) * att_loss
        return (loss, ctc_loss.item(), att_loss.item())
        
        
class CTC_Loss(nn.Module):
    def __init__(
        self,
        blank_id = None,        
    ):
        super().__init__()
        self.ctc   = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        
    def forward(self, outputs, targets, target_lengths, att=None, *args, **kwargs):
        ctc_out = outputs.contiguous().permute(1,0,2) # (B,L,E)->(L,B,E)
        ctc_loss = self.ctc(ctc_out, targets,
                            (torch.ones(ctc_out.shape[1])*ctc_out.shape[0]).to(torch.int),
                            target_lengths,)
        return ctc_loss
    
        
class Attention_Loss(nn.Module):
    def __init__(
        self, 
        ignore_index : int = None,
        label_smoothing : float = 0.0,
    ):
        super().__init__()
        self.att = nn.CrossEntropyLoss(
            reduction='mean', 
            ignore_index=ignore_index,
        )
        
    def forward(self, outputs, targets, att=None, *args, **kwargs):
        out = outputs.contiguous().view(-1,outputs.shape[-1])
        targets = targets.contiguous().view(-1)
        loss = self.att(out, targets)
        return loss

