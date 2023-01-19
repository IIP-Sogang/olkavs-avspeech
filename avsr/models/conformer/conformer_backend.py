import torch
import torch.nn as nn

from avsr.models.conformer.encoder import ConformerEncoder


class Conformer_back(nn.Module):
    '''
    input :  (BxLxD)
    output : (BxLxD)
    '''
    def __init__(
        self,
        encoder_n_layer : int,
        encoder_d_model : int, 
        encoder_n_head : int, 
        encoder_ff_dim : int,
        encoder_dropout_p : float,
    ):
        super().__init__()
        self.conformer = ConformerEncoder(encoder_d_model, encoder_d_model, 
                                          encoder_n_layer, encoder_n_head, 
                                          input_dropout_p=encoder_dropout_p)
        self.layers = nn.Sequential(*self.conformer.layers)
        
    def forward(self, inputs, input_lengths):
        outputs = self.layers(inputs)
        return outputs