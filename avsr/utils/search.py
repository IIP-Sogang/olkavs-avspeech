import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
mp = mp.get_context('spawn')

from ..models.model import *
from ..models.encoder import *
from ..models.decoder import *

EPSILON = 1e-300


def end_detect(hypothesis, length, M=12, threshold=-4, progressive=None):
    _, max_score, _, _, _ = max(hypothesis, key=lambda x: x[1])
    _, max_l, _, _, _ = max(hypothesis[-M:], key=lambda x: x[1])

    if progressive:
        _, beam_max_score, _, _, _ = max(progressive, key=lambda x: x[1])
        max_l = max(max_l, beam_max_score)
    
    if max_l - max_score < threshold:
        return True
    else:
        return False
 

class SearchSequence:
    def __init__(
        self,
        model,
        vocab_size:int,
        method:str='hybrid',
        pad_id:int=0,
        sos_id:int=1,
        eos_id:int=2,
        unk_id:int=3,
        max_len : int = 150,
        ctc_rate:float=0.3,
        mp_num:int=None,
    ):
        self.model = model
        self.ctc_decoder = self.model.decoder.ctcdecoder if 'ctcdecoder' in dir(self.model.decoder) else self.model.decoder
        self.att_decoder = self.model.decoder.attdecoder if 'attdecoder' in dir(self.model.decoder) else self.model.decoder
        self.embedder = self.model.embedder
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.blank_id = unk_id

        self.max_len = max_len

        self.method = method
        if self.method=='ctc':
            self.ctc_rate = 1
        elif self.method in ['att']:
            self.ctc_rate = 0
        else:
            self.ctc_rate = ctc_rate

        self.mp_num = mp_num

    def __call__(
        self,
        video_inputs,
        video_input_lengths,
        audio_inputs,
        audio_input_lengths,
        beam_size : int = 1,
        D_end : int = -4,
        M_end : int = 12,
        device = 'cpu',
        mp_num : int = 1,
        *args, **kwargs
    ):
        batch_size = video_inputs.size(0)

        hypothesis = torch.full((batch_size, self.max_len), self.pad_id).to(device)
        max_lengths = torch.zeros(batch_size, dtype=int).to(device)

        hypothesis.share_memory_()
        max_lengths.share_memory_()

        self._search_batch(
            video_inputs,
            video_input_lengths,
            audio_inputs,
            audio_input_lengths,
            beam_size,
            D_end,
            M_end,
            device,
            mp_num,
            [hypothesis, max_lengths]
        )
        return hypothesis, max_lengths

    def _search_batch(
        self,
        video_inputs,
        video_input_lengths,
        audio_inputs,
        audio_input_lengths,
        beam_size : int = 1,
        D_end : int = -4,
        M_end : int = 12,
        device = 'cpu',
        mp_num: int= 1,
        shared_outputs: List= None,
        *args, **kwargs
    ):
        batch_size = video_inputs.size(0)
        features = self.model.encoder(video_inputs, video_input_lengths,
                                audio_inputs, audio_input_lengths)
        features = self.model.medium(features)
        
        self._search(features, beam_size, D_end, M_end, device, 0, batch_size, shared_outputs)
        
    def _search(
        self,
        features = None,
        beam_size : int = 1,
        D_end : int = -4,
        M_end : int = 12,
        device = 'cpu',
        batch_idx = None,
        batch_size = 1,
        shared_outputs = None,
        *args, **kwargs
    ):
        # sos sequence
        y_hats = torch.full((batch_size, self.max_len), self.pad_id).to(device)
        y_hats[:,0] = self.sos_id
        
        # length-limited hypothesis, initialized 'hypo_0'
        hypo_l = [[(y_hats[i],0,0,0,0)] for i in range(batch_size)]
        
        # completed hypothesis
        hypothesis = [[] for _ in range(batch_size)]
        
        # flag
        skip = np.array([False for _ in range(batch_size)])
        
        if self.method in ['hybrid', 'ctc']:
            ctc_output = self._init_ctc(features)
        else:
            ctc_output = None

        # search...
        for length in range(1, self.max_len):
            active = np.arange(batch_size)[~skip]
            hypo_sub, beam = self._beam_search(
                features[active],
                [hypo_l[i] for i in active],
                length,
                batch_size - skip.sum(),#batch_size,
                beam_size,
                ctc_output[active],
                device
            )
            
            for i in active:
                # Add complete hypothesis
                hypothesis[i].append(max(hypo_sub[i], key=lambda x: x[1]))
                #end detect
                if end_detect(hypothesis[i], length, M_end, D_end, progressive=beam[i]):
                    skip[i] = True
            if skip.sum()==batch_size:
                break
            
            hypo_l = beam
        
        max_hypothesis = torch.zeros_like(y_hats)
        max_lengths = torch.zeros(batch_size, dtype=int)
        for i in range(batch_size):
            _max_hypothesis, max_score, _length, _, _ = max(hypothesis[i], key=lambda x: x[1])
            max_hypothesis[i] = _max_hypothesis
            max_lengths[i] = _length
        
        # return to Shared memory
        cur_idx = batch_idx * batch_size
        shared_outputs[0][cur_idx: cur_idx+batch_size, :length-1] = max_hypothesis[:, 1:length]
        shared_outputs[1][cur_idx: cur_idx+batch_size] = max_lengths

        return max_hypothesis[:, 1:length], max_lengths

    def _beam_search(
        self,
        features,
        current_hypothesis,
        length:int,
        batch_size:int,
        beam_size:int,
        ctc_output=None,
        device='cpu',
    ):
        # complete sub hypothesis
        hypo_sub = [[] for _ in range(batch_size)]
    
        # make beam
        beam = [[] for _ in range(batch_size)]
        min_score = [-float('inf') for _ in range(batch_size)]

        # token control
        unused_tokens = [self.pad_id, self.eos_id, self.sos_id]
        token_mask = ~np.isin(np.arange(self.vocab_size),unused_tokens)

        # check active beam
        while current_hypothesis[0]:
            # get predictions for each batch from beams
            g = torch.zeros((batch_size, self.max_len), dtype=int).to(device)
            g_att = np.zeros((batch_size,), dtype=float)
            for i in range(batch_size):
                _g, g_score, g_len, _g_att, g_ctc = current_hypothesis[i].pop(0)
                g[i] = _g
                g_att[i] = _g_att

            scores, ctc_scores, att_scores = self.get_scores(
                                                        features=features,
                                                        seq_length=length,
                                                        g=g,
                                                        g_att=g_att,
                                                        ctc_output=ctc_output,
                                                )

            y_hat = g.detach().clone()
            y_hat[:, length] = self.eos_id
            
            for i in range(batch_size):
                hypo_sub[i].append((y_hat[i], scores[i,self.eos_id].item(), length, att_scores[i,self.eos_id].item(), ctc_scores[i,self.eos_id].item()))
            
                # if "room exists for a new hypo"
                # Fill beams with initial values
                token = 0
                while len(beam[i]) < beam_size:
                    if token in unused_tokens: pass
                    else: beam[i].append((y_hat[i], scores[i,token].item(), length, att_scores[i,token].item(), ctc_scores[i,token].item()))
                    token += 1
                beam[i] = sorted(beam[i], key=lambda x: x[1], reverse=True) # Sort by score, descending

                # "Hypo of which score is higher than beam's min score, except special tokens."
                while (scores[i, token_mask]>min_score[i]).sum():
                    # get one token from condition
                    token = np.arange(self.vocab_size)[(scores[i]>min_score[i]) * token_mask][0]
                    y_hat[i] = y_hat[i].clone()
                    y_hat[i, length] = int(token)
                    beam[i].append((y_hat[i], scores[i,token].item(), length, att_scores[i,token].item(), ctc_scores[i,token].item()))
                    beam[i] = sorted(beam[i], key=lambda x: x[1], reverse=True) # Sort by score, descending
                    beam[i].pop(-1)
                    min_score[i] = beam[i][-1][1]

        return hypo_sub, beam

    def _init_ctc(
        self,
        features,
    ):
        batch_size = features.size(0)

        # get decoders
        if 'ctcdecoder' in dir(self.model.decoder):
            ctc_decoder = self.model.decoder.ctcdecoder
        else:
            ctc_decoder = self.model.decoder
        ctc_output = F.softmax(ctc_decoder(inputs=features), dim=-1)
        self.ctc_len = ctc_output.size(1)
        ctc_output = F.pad(ctc_output, (0,0,1,0))

        # initialize score_dictionary for dynamic ctc score calculation
        self.y_n:List[Dict[str,float]] = [[{str(self.sos_id):0} for t in range(self.ctc_len+1)] for _ in range(batch_size)]
        self.y_b:List[Dict[str,float]] = [[{str(self.sos_id):1}] for _ in range(batch_size)]
        for i in range(batch_size):
            for t in range(self.ctc_len):
                self.y_b[i].append({str(self.sos_id):self.y_b[i][t][str(self.sos_id)] * ctc_output[i,t+1,self.blank_id].item()})
        return ctc_output

    def get_scores(
        self,
        features,
        seq_length:int,
        g,
        g_att:List[float]=None,
        ctc_output=None,
    ):
        batch_size = g.shape[0]
        ctc_scores = np.full((batch_size, self.vocab_size), -float('inf'))
        att_scores = np.full((batch_size, self.vocab_size), -float('inf'))
        
        if self.method != 'ctc':
            # calculate attention score
            y_emb = self.embedder(F.one_hot(g[:,:seq_length], num_classes=self.vocab_size).to(torch.float32))
            att_scores = F.log_softmax(self.att_decoder(inputs=features, labels=y_emb, pad_id=self.pad_id), dim=-1)
            att_scores = att_scores.detach().cpu().numpy()
            att_scores = att_scores[:,-1] + g_att.reshape(-1,1)
        
        if self.method != 'att':
            ctc_scores = np.zeros((batch_size, self.vocab_size))
            for i in range(batch_size):
                ctc_scores[i] = ctc_label_scores(
                                    " ".join(map(lambda x:str(x.item()), g[i,:seq_length])), self.vocab_size, 
                                    X = ctc_output[i].detach().cpu().numpy(), 
                                    T = self.ctc_len,
                                    y_n = self.y_n[i],
                                    y_b = self.y_b[i],
                                    max_len=self.max_len, 
                                    sos_id=self.sos_id, 
                                    eos_id=self.eos_id, 
                                    pad_id=self.pad_id,
                                    blank_id=self.blank_id,
                                )
            
        scores = self.ctc_rate * ctc_scores + (1-self.ctc_rate) * att_scores

        return scores, ctc_scores, att_scores

def ctc_label_scores(
    g: str,
    vocab_size: int,
    X,
    T: int,
    y_n: Dict[int, Dict[str, float]],
    y_b: Dict[int, Dict[str, float]],
    max_len=150,
    sos_id=None,
    eos_id=None,
    blank_id=None,
    pad_id=None,
    last_score=None,
):
    """
    Compute the CTC label scores for each hypothesized character in a vectorized manner.

    Args:
        g: string representation of the previous labels in the sequence.
        vocab_size: size of the vocabulary.
        X: CTC probabilities matrix.
        T: Number of time steps in the input sequence.
        y_n: non-blank probability dictionary.
        y_b: blank probability dictionary.
        max_len: maximum length of the sequence (default 150).
        sos_id: start of sequence label ID.
        eos_id: end of sequence label ID.
        blank_id: blank label ID.
        pad_id: padding label ID.
        last_score: last computed score (optional).

    Returns:
        The CTC label scores (log probabilities) for each hypothesized character.
    """
    last_token = int(g.split()[-1])
    vocabs: List[int] = [
        c for c in range(vocab_size) if c not in [sos_id, eos_id, pad_id]
    ]

    yn = np.zeros(vocab_size)
    yn[:] = X[1] if g == str(sos_id) else 0
    yb = np.zeros(vocab_size)

    valid_keys = [join_key(g, c) for c in vocabs]

    for key in valid_keys:
        y_n[1][key] = yn[int(key.split()[-1])]
        y_b[1][key] = 0

    psi = yn.copy()
    phi = np.zeros(vocab_size)

    for t in range(2, T + 1):
        yn_prev = yn.copy()
        yb_prev = yb.copy()

        phi[:] = y_b[t - 1][g] + y_n[t - 1][g]
        phi[last_token] = y_b[t - 1][g]

        yn = (yn_prev + phi) * X[t]
        for key in valid_keys:
            y_n[t][key] = yn[int(key.split()[-1])]

        yb = (yb_prev + yn_prev) * X[t][blank_id]
        for key in valid_keys:
            y_b[t][key] = yb[int(key.split()[-1])]

        psi += phi * X[t]
    psi[eos_id] = y_n[T][g] + y_b[T][g]
    scores = np.log(psi + EPSILON)

    return scores


def join_key(a: str, b: int):
    return a + " " + str(b)
