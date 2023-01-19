import torch
import torch.nn as nn

from ..models.model import *
from ..models.encoder import *
from ..models.decoder import *
from ..models.medium import MLPLayer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    vocab_size : int, pad_id : int, 
    architecture : str = 'audio_visual', loss_fn : str = 'hybrid', front_dim=None, 
    encoder_n_layer=None,  encoder_d_model=None, 
    encoder_n_head=None, encoder_ff_dim=None, encoder_dropout_p=None,
    decoder_n_layer=None, decoder_d_model=None, 
    decoder_n_head=None, decoder_ff_dim=None,  decoder_dropout_p=None,
    pass_visual_frontend = True, verbose = True):

    if verbose:
        print(f"Build {loss_fn} {architecture} model...")    

    # Define Model Class
    if loss_fn == 'hybrid':
        Model = HybridModel
    elif loss_fn == 'att' or architecture=='attention':
        Model = AttentionModel
    elif loss_fn == 'ctc':
        Model = CTCModel

    # Define Encoder Class
    if architecture == 'audio_visual': 
        Encoder = FusionConformerEncoder
        num_modalities = 2
    elif architecture == 'audio' : 
        Encoder = AudioConformerEncoder
        num_modalities = 1
    elif architecture == 'video' :
        Encoder = VisualConformerEncoder
        num_modalities = 1

    # Define Decoder Class
    if loss_fn == 'hybrid':
        Decoder = HybridDecoder
    elif loss_fn == 'att':
        Decoder = TransformerDecoder
    elif loss_fn == 'ctc':
        Decoder = LinearDecoder
    

    # Define Arguments
    encoder_kwargs = dict(
        front_dim = front_dim,
        encoder_n_layer=encoder_n_layer, 
        encoder_d_model=encoder_d_model,
        encoder_n_head=encoder_n_head, 
        encoder_ff_dim=encoder_ff_dim, 
        encoder_dropout_p=encoder_dropout_p,
        pass_visual_frontend=pass_visual_frontend)

    decoder_kwargs = dict(
        vocab_size = vocab_size,
        decoder_n_layer=decoder_n_layer, 
        decoder_d_model=decoder_d_model,
        decoder_n_head=decoder_n_head, 
        decoder_ff_dim=decoder_ff_dim, 
        decoder_dropout_p=decoder_dropout_p)


    model = Model(vocab_size=vocab_size, pad_id=pad_id)
    model.embedder = nn.Linear(vocab_size, decoder_d_model)
    model.encoder = Encoder(**encoder_kwargs)
    model.medium = MLPLayer(encoder_d_model*num_modalities, encoder_d_model)
    model.decoder = Decoder(**decoder_kwargs)

    if verbose:
        print("Build complete.")
        print(f"# of total parameters : {count_parameters(model)}")

    return model