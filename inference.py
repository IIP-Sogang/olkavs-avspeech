import yaml
import time
import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from avsr.utils.getter import get_metric, select_search
from avsr.utils.model_builder import build_model
from vocabulary.utils import KsponSpeechVocabulary
from dataset.dataset import *

mp = mp.get_context('spawn')


def show_description(it, total_it, ger, mean_ger, cer, mean_cer, wer, mean_wer, swer, mean_swer, _time):
    train_time = int(time.time() - _time)
    _sec = train_time % 60
    train_time //= 60
    _min = train_time % 60
    train_time //= 60
    _hour = train_time % 24
    _day = train_time // 24
    desc = f"GER {ger:.4f} :: MEAN GER {mean_ger:.4f} :: CER {cer:.4f} :: MEAN CER {mean_cer:.4f} :: WER {wer:.4f} :: MEAN WER {mean_wer:.4f} :: sWER {swer:.4f} :: MEAN sWER {mean_swer:.4f} :: BATCH [{it}/{total_it}] :: [{_day:2d}d {_hour:2d}h {_min:2d}m {_sec:2d}s]"
    print(desc, end="\r")


def load_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(f"{checkpoint_path}", map_location=device)
    model.load_state_dict(state_dict, strict=False)

def single_infer(config, vocab, search, metrics, audio_transform, tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts, device='cuda'):

    vids = _parse_video(tr_video_paths, is_raw=config["raw_video"])
    signal, _ = get_sample(tr_audio_paths,resample=config['audio_sample_rate'])
    seqs = _parse_audio(signal, audio_transform, config['audio_normalize'])
    targets = _parse_transcript(tr_transcripts, vocab.sos_id, vocab.eos_id)
    # korean_transcript = _parse_korean_transcript(tr_korean_transcripts, vocab.sos_id, vocab.eos_id)

    if config["raw_video"]:
        # B T W H C --> B C T W H
        vids = vids.permute(3,0,1,2)
    seqs = seqs.permute(1,0)

    vids = vids.unsqueeze(0).to(device)
    seqs = seqs.unsqueeze(0).to(device)
    targets = targets.unsqueeze(0).to(device)
    vid_lengths = torch.full((1,),vids.size(1)).to(int)
    seq_lengths = torch.full((1,),seqs.size(1)).to(int)
    target_lengths = torch.full((1,),targets.size(1)).to(int)
    
    with torch.no_grad():
        outputs, output_lengths = search(
            vids, vid_lengths,
            seqs, seq_lengths,
            device = device,
            beam_size = config['beam_size'],
            D_end = config['EndDetect_D'],
            M_end = config['EndDetect_M'],
        )

    path_units = tr_video_paths.strip().split('/')
    file_index = ",".join([path_units[-2], path_units[-1].replace(".npy","")])

    errorRates = list()
    for metric in metrics:
        errorRates.append(metric(outputs=outputs, targets=targets[:,1:], output_lengths=output_lengths, target_lengths=target_lengths, file_path=file_index))
    
    return errorRates

def infer(config, model, vocab, dataset, scores, device='cpu'):
     # Assertion
    assert config['num_mp'] <= len(dataset), "num_mp should be equal or smaller than size of dataset!!"

    # define a criterion
    metric_ger = get_metric(vocab, config['script_path'], unit=config['tokenize_unit'], error_type='ger')
    metric_cer = get_metric(vocab, config['script_path'], unit=config['tokenize_unit'], error_type='cer')
    metric_wer = get_metric(vocab, config['script_path'], unit=config['tokenize_unit'], error_type='wer')
    metric_swer = get_metric(vocab, config['script_path'], unit=config['tokenize_unit'], error_type='swer')
    metrics = [metric_ger, metric_cer, metric_wer, metric_swer]

    search = select_search(
        model = model,
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        unk_id=vocab.unk_id,
        method = config['search_method'],
        max_len=config['max_len'],
        ctc_rate=config['ctc_rate'],
        mp_num=config['num_mp']
    )

    model.eval()
    eval_start = time.time()
    
    # define a loader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn = AVcollator(
            max_len = config['max_len'],
            use_video = config['use_video'],
            raw_video = config['raw_video'],
            infer = True), 
        shuffle = False,
        num_workers=config['num_workers'],
    )

    for it, (vids, seqs, targets, vid_lengths, seq_lengths, target_lengths, paths) in enumerate(dataloader):
        vids = vids.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        vid_lengths = vid_lengths.to(device)
        seq_lengths = seq_lengths.to(device)
        
        with torch.no_grad():
            outputs, output_lengths = search(
                vids, vid_lengths,
                seqs, seq_lengths,
                device = device,
                beam_size = config['beam_size'],
                D_end = config['EndDetect_D'],
                M_end = config['EndDetect_M'],
                mp_num = config['num_mp'],
            )
        path_units = [tr_video_path.strip().split('/') for tr_video_path in paths]
        file_indices = [",".join([path_unit[-2], path_unit[-1].replace(".npy","")]) for path_unit in path_units]

        errorRates = list()
        for metric in metrics:
            errorRates.append(metric(outputs=outputs, targets=targets[:,1:], 
                                     output_lengths=output_lengths, target_lengths=target_lengths, 
                                     file_path=file_indices))
        
        ger, cer, wer, swer = errorRates
        scores += torch.tensor([1, ger, cer, wer, swer])
        
        
        # show description
        show_description(
            it = int(scores[0].item()),
            total_it = len(dataloader),
            ger = ger,
            mean_ger = scores[1]/scores[0],
            cer = cer,
            mean_cer = scores[2]/scores[0],
            wer = wer,
            mean_wer = scores[3]/scores[0],
            swer = swer,
            mean_swer = scores[4]/scores[0],
            _time = eval_start
        )


def main(args, loop=None):
    
    # Check Devices
    print("cuda : ", torch.cuda.is_available())
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    vocab = KsponSpeechVocabulary(unit = config['tokenize_unit'])
    
    # fix the seed
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    # load dataset
    dataset = prepare_dataset(
        transcripts_path = config['transcripts_path_test'],
        vocab = vocab,
        use_video = config['use_video'],
        raw_video = config['raw_video'],
        audio_transform_method = config['audio_transform_method'],
        audio_sample_rate = config['audio_sample_rate'],
        audio_normalize = config['audio_normalize'],
        return_path=True
    )

    print(f"# of data : {len(dataset)}")
    print('Batch size :',config['batch_size'])
   
    scores = torch.tensor([0.,0.,0.,0.,0.]) # num, ger, cer, wer, swer
    scores.share_memory_()
    
    # define a model
    model = build_model(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        architecture=config['architecture'],
        loss_fn=config['loss_fn'],
        front_dim=config['front_dim'],
        encoder_n_layer=config['encoder_n_layer'],
        encoder_d_model=config['encoder_d_model'],
        encoder_n_head=config['encoder_n_head'], 
        encoder_ff_dim=config['encoder_ff_dim'], 
        encoder_dropout_p=config['encoder_dropout_p'],
        decoder_n_layer=config['decoder_n_layer'],
        decoder_d_model=config['decoder_d_model'],
        decoder_n_head=config['decoder_n_head'], 
        decoder_ff_dim=config['decoder_ff_dim'], 
        decoder_dropout_p=config['decoder_dropout_p'],
        pass_visual_frontend= not config['raw_video'],
        verbose=True
    )
    # load state dict
    load_checkpoint(model, checkpoint_path=config['model_path'], device=DEVICE)
    # move the model to GPU
    model.to(DEVICE)

    infer(config, model, vocab, dataset, scores, DEVICE)

    print()
    print()
    print("Warning : This Results Do Represent Only the Estimation of Error Rates.")
    print("          Re-calculation of Error Rates Should be Done, by Considering the Lengths of Sequence")
    print(f"[Results]")
    print(f"Grapheme Error Rate  : {100*scores[1]/scores[0]:.2f}%")
    print(f"Character Error Rate : {100*scores[2]/scores[0]:.2f}%")
    print(f"Word Error Rate      : {100*scores[3]/scores[0]:.2f}%")
    print(f"sWord Error Rate     : {100*scores[4]/scores[0]:.2f}%")
    
    with open('results/inference_log/'+config['log_path']+'.txt', 'a') as f:
        import datetime
        f.write(f"""
        {datetime.datetime.today()}
        Evalation Results of {config['model_path']} for {config['transcripts_path_test']}.
        Search method : {config['search_method']}, CTC_rate : {config['ctc_rate']}
        Beam size : {config['beam_size']}
        Grapheme Error Rate  : {100*scores[1]/scores[0]:.2f}%
        Character Error Rate : {100*scores[2]/scores[0]:.2f}%
        Word Error Rate      : {100*scores[3]/scores[0]:.2f}%
        sWord Error Rate     : {100*scores[4]/scores[0]:.2f}%
        """)


def get_args():
    parser = argparse.ArgumentParser(description='option for AV training.')
    parser.add_argument('-w','--world_size',
                         default=0, type=int, help='Configurate the number of GPUs')
    parser.add_argument('-c','--config',
                         type=str, help='Configuration Path')
    parser.add_argument('-p','--port',
                         default = '12355', type = str, help='Port number of multi process')
    args = parser.parse_args()
    return args
  

if __name__ == '__main__':
    args = get_args()
    main(args)