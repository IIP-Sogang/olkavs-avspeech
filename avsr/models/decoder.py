import pdb
import re
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridDecoder(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        decoder_n_layer : int,
        decoder_d_model : int, 
        decoder_n_head : int, 
        decoder_ff_dim : int, 
        decoder_dropout_p : float,
    ):
        super().__init__()
        self.attdecoder = TransformerDecoder(
            vocab_size=vocab_size,
            decoder_n_layer=decoder_n_layer,
            decoder_d_model=decoder_d_model, 
            decoder_n_head=decoder_n_head, 
            decoder_ff_dim=decoder_ff_dim, 
            decoder_dropout_p=decoder_dropout_p,
        )
        self.ctcdecoder = LinearDecoder(vocab_size=vocab_size, decoder_d_model=decoder_d_model)
    
    def forward(self, labels, inputs, pad_id=None, **kwargs):
        output_a = self.attdecoder(inputs=inputs, labels=labels, pad_id=pad_id)
        output_b = self.ctcdecoder(inputs=inputs)
        return (output_a, output_b) # (B, M, D)


class TransformerDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(
        self,
        vocab_size : int,
        decoder_n_layer : int,
        decoder_d_model : int, 
        decoder_n_head : int, 
        decoder_ff_dim : int, 
        decoder_dropout_p : float,
    ):
        super().__init__()
        decoder = nn.TransformerDecoderLayer(decoder_d_model, decoder_n_head, 
                                             decoder_ff_dim, decoder_dropout_p)
        self.decoder = nn.TransformerDecoder(decoder, decoder_n_layer)
        self.fc = nn.Linear(decoder_d_model, vocab_size)
        
    def forward(self, labels, inputs, pad_id=None, **kwargs):
        label_mask = self.generate_square_subsequent_mask(labels.shape[1]).to(inputs.device)
        label_pad_mask = self.get_attn_pad_mask(torch.argmax(labels, dim=-1), pad_id) if pad_id else None
        
        labels = labels.permute(1,0,2)
        inputs = inputs.permute(1,0,2)
        
        outputs = self.decoder(labels, inputs, 
                               tgt_mask=label_mask,
                               tgt_key_padding_mask=label_pad_mask)
        
        outputs = outputs.permute(1,0,2)
        outputs = self.fc(outputs)
        return outputs

    def score(self,
        text:str,
        features:torch.Tensor,
        embedder:torch.nn.Module,
        vocab2idx:dict=None,
        pad_id:int=0,
        device:str='cuda',
    ):
        text = '<sos>'+text
        g = self._text2vec(text, vocab2idx, pad_id=pad_id).to(device)
        y_emb = embedder.to(device)(F.one_hot(g, num_classes=len(vocab2idx)).to(torch.float32))
        att_scores = F.log_softmax(self.forward(inputs=features, labels=y_emb, pad_id=pad_id), dim=-1)
        att_score = att_scores[0,-1]
        att_score[pad_id] = -float('inf')
        
        return att_score.tolist()

    def _text2vec(self, 
        texts, 
        vocab2idx:Dict[str,int],
        pad_id:int = 0,
        return_length:bool = False,
    ):
        if isinstance(texts, list):
            vectors = []
            for text in texts:
                vec = []
                token = ""
                for chr in text:
                    token += chr
                    if chr == '\\' or re.match("[<a-zA-Z]", chr):
                        continue
                    vec.append(vocab2idx[token])
                    token = ""
                vectors.append(vec)
            
            max_length = len(max(vectors, key=lambda x: len(x)))
            vector = torch.full((len(texts), max_length), pad_id)
            vector_lengths = torch.zeros(len(vectors), dtype=int)

            for i,vec in enumerate(vectors):
                vector[i, :len(vec)] = torch.tensor(vec)
                vector_lengths[i] = len(vec)
        
        else:
            vec = []
            token = ""
            for chr in texts:
                token += chr
                if chr == '\\' or re.match("[<a-zA-Z]", chr):
                    continue
                vec.append(vocab2idx[token])
                token = ""
            vector = torch.tensor([vec])
            vector_lengths = torch.tensor([len(vec)])

        if return_length:
            return vector, vector_lengths
        else: 
            return vector

    def generate_square_subsequent_mask(self, sz):
        mask = torch.full((sz, sz),-float('inf'))
        mask = torch.triu(mask, diagonal=1)
        return mask
        
    def get_attn_pad_mask(self, seq, pad):
        batch_size, len_seq = seq.size()
        pad_attn_mask = seq.eq(pad)
        return pad_attn_mask
        
        
class LinearDecoder(nn.Module):
    '''
    Inputs : (B x S x E), (B x T x E)
    '''
    def __init__(
        self, 
        vocab_size : int,
        decoder_d_model : int, 
        *args,
        **kwargs
    ):
        super().__init__()
        self.fc = nn.Linear(decoder_d_model, vocab_size)
        
    def forward(self, inputs, pad_id=None, **kwargs):
        outputs = self.fc(inputs)
        return outputs
        