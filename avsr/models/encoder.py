import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .conformer.conformer_backend import Conformer_back
from .resnet.resnet import Resnet1D_front, Resnet2D_front

class FusionConformerEncoder(nn.Module):
    def __init__(
            self, front_dim : int, encoder_n_layer : int, 
            encoder_d_model : int, encoder_n_head : int, 
            encoder_ff_dim : int, encoder_dropout_p : float,
            pass_visual_frontend : bool = True):
        super().__init__()
        self.audio_model = AudioConformerEncoder(
            front_dim = front_dim,
            encoder_n_layer=encoder_n_layer, encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p)
        self.visual_model = VisualConformerEncoder(
            front_dim = front_dim, encoder_n_layer=encoder_n_layer, 
            encoder_d_model=encoder_d_model, encoder_n_head=encoder_n_head, 
            encoder_ff_dim=encoder_ff_dim, encoder_dropout_p=encoder_dropout_p,
            pass_visual_frontend = pass_visual_frontend)
        
    def forward(self, 
            video_inputs, video_input_lengths,
            audio_inputs, audio_input_lengths,
            *args, **kwargs):
        audio_feature = self.audio_model(None, None, audio_inputs, audio_input_lengths)
        visual_feature = self.visual_model(video_inputs, video_input_lengths, None, None)
        features = self.fusion(visual_feature, audio_feature)
        return features
        
    def fusion(self, visual_feature, audio_feature):
        diff = audio_feature.size(1) - visual_feature.size(1)
        front_margin = diff//2
        back_margin = diff - front_margin
        visual_feature = F.pad(visual_feature, (0, 0, front_margin, back_margin), 'constant', 0)
    
        outputs = torch.cat([visual_feature, audio_feature], dim=-1)
        
        return outputs


class AudioConformerEncoder(nn.Module):
    def __init__(
            self, front_dim : int, encoder_n_layer : int, 
            encoder_d_model : int, encoder_n_head : int, 
            encoder_ff_dim : int, encoder_dropout_p : float,
            *args, **kwargs):
        super().__init__()
        self.front = Resnet1D_front(1, front_dim,)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, encoder_d_model=encoder_d_model,
            encoder_n_head=encoder_n_head, encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p)
        self.projection = nn.Linear(front_dim, encoder_d_model)
        
    def forward(
            self, 
            video_inputs, video_input_lengths,
            audio_inputs, audio_input_lengths,
            *args, **kwargs):
        outputs = self.front(audio_inputs)
        outputs = outputs.permute(0,2,1) # (B, L, C)
        outputs = self.projection(outputs)
        outputs = self.back(outputs, audio_input_lengths)
        return outputs


class VisualConformerEncoder(nn.Module):
    def __init__(
            self, front_dim : int, encoder_n_layer : int,  encoder_d_model : int, 
            encoder_n_head : int,  encoder_ff_dim : int,  encoder_dropout_p : float,
            pass_visual_frontend : bool = True):
        super().__init__()
        self.pass_front = pass_visual_frontend
        if self.pass_front: pass
        else: self.front = Resnet2D_front(3, front_dim,)
        self.back  = Conformer_back(
            encoder_n_layer=encoder_n_layer, encoder_d_model=encoder_d_model, 
            encoder_n_head=encoder_n_head, encoder_ff_dim=encoder_ff_dim, 
            encoder_dropout_p=encoder_dropout_p)
        self.projection = nn.Linear(front_dim, encoder_d_model)
        
    def forward(
            self, 
            video_inputs, video_input_lengths,
            audio_inputs = None, audio_input_lengths = None,
            *args, **kwargs):
        if self.pass_front: outputs = video_inputs
        else: outputs = self.front(video_inputs)
        outputs = self.projection(outputs)
        outputs = self.back(outputs, video_input_lengths)
        return outputs
