import os
import pdb
import csv
import sys
import math
import random
import platform
import warnings

import torch
import torchaudio
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile")
elif platform.system() == 'Linux':
    torchaudio.set_audio_backend("sox_io")
from torch import Tensor, FloatTensor
from torch.utils.data import Dataset

import numpy as np
from numpy.lib.stride_tricks import as_strided

from vocabulary.utils import Vocabulary
from dataset.augment import SpecAugment, BabbleNoise, get_sample
from dataset.feature import MelSpectrogram,MFCC,Spectrogram,FilterBank


def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels
    Args:
        transcripts_path (str): path of transcripts
    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    video_paths = list()
    audio_paths = list()
    korean_transcripts = list()
    transcripts = list()

    with open(transcripts_path, encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            # pdb.set_trace()
            video_path, audio_path, korean_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')
            video_paths.append(video_path)
            audio_paths.append(audio_path)
            korean_transcripts.append(korean_transcript)
            transcripts.append(transcript)

    return video_paths, audio_paths, korean_transcripts, transcripts


def prepare_dataset(
    transcripts_path: str,
    vocab: Vocabulary,
    use_audio: bool = True,
    use_video: bool = True,
    raw_video: bool = True,
    audio_transform_method = 'fbank',
    audio_sample_rate = None,
    audio_n_mels = None,
    audio_frame_length = None,
    audio_frame_shift = None,
    audio_normalize = False,
    spec_augment: bool = False,
    freq_mask_para: int = None,
    freq_mask_num: int = None,
    time_mask_num: int = None,
    noise_rate: float = 0.,
    noise_path: str = None,
    Train=True,
    return_path: bool = False,
):

    train_or_test = 'train' if Train else 'test'
    print(f"prepare {train_or_test} dataset start !!")

    tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts = load_dataset(transcripts_path)
    print("Loaded dataset from", transcripts_path)

    trainset = AV_Dataset(
            video_paths=tr_video_paths,
            audio_paths=tr_audio_paths,
            korean_transcripts=tr_korean_transcripts,
            transcripts=tr_transcripts,
            sos_id=vocab.sos_id, 
            eos_id=vocab.eos_id,
            use_audio=use_audio,
            use_video = use_video,
            raw_video = raw_video,
            audio_transform_method = audio_transform_method,
            audio_sample_rate = audio_sample_rate,
            audio_n_mels = audio_n_mels,
            audio_frame_length = audio_frame_length,
            audio_frame_shift = audio_frame_shift,
            audio_normalize = audio_normalize,
            spec_augment = spec_augment,
            freq_mask_para = freq_mask_para,
            freq_mask_num = freq_mask_num,
            time_mask_num = time_mask_num,
            noise_rate = noise_rate,
            noise_path = noise_path,
            return_path=return_path,
    )
    
    print(f"prepare {train_or_test} dataset finished.")

    return trainset

def _parse_video(video_path, is_raw=False):
    if is_raw:
        import cv2
        assert os.path.exists(video_path), f"{video_path} does not exist!"
        frames = []
        cap = cv2.VideoCapture(video_path)
        ret = True
        while ret:
            ret, img = cap.read()
            if ret:
                frames.append(img)
        video = np.stack(frames, axis=0) # T,H,W,C
        video = torch.from_numpy(video).float()
        video -= torch.mean(video)
        video /= torch.std(video)
    else:
        video = np.load(video_path)
        video = torch.from_numpy(video).float()
        
    return video

def _parse_audio(signal, transform, normalize):
    signal = signal.numpy().reshape(-1,)    
    feature = transform(signal)
    if normalize:
        feature -= feature.mean()
        feature /= np.std(feature)
    return FloatTensor(feature)

def _parse_transcript(transcript, sos_id, eos_id):
    tokens = transcript.strip().split(' ')
    transcript = list()
    transcript.append(int(sos_id))
    for token in tokens:
        transcript.append(int(token))
    transcript.append(int(eos_id))
    return torch.tensor(transcript)

def _parse_korean_transcript(korean_transcript, sos_id, eos_id):
    tokens = korean_transcript.split(' ')
    korean_transcript = list()
    korean_transcript.append(str(sos_id))
    for token in tokens:
        korean_transcript.append(str(token))
    korean_transcript.append(str(eos_id))
    return korean_transcript

def _prepare_dataset(
    transcripts_path: str,
):
    tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts = load_dataset(transcripts_path)
    return list(zip(tr_video_paths, tr_audio_paths, tr_korean_transcripts, tr_transcripts))

class AV_Dataset(Dataset):
    # Augmentation Index
    # Not apply augmentation
    # 1 : SpecAugment, 2: NoiseAugment, 3: Both
    VANILLA = 0
    SPEC_AUGMENT = 1
    NOISE_AUGMENT = 2
    BOTH_AUGMENT = 3

    def __init__(
            self,
            video_paths: list,              # list of video paths
            audio_paths: list,              # list of audio paths
            korean_transcripts: list,
            transcripts: list,              # list of transcript paths
            sos_id: int,                    # identification of start of sequence token
            eos_id: int,                    # identification of end of sequence token
            use_audio: bool = True,
            use_video: bool = True,
            raw_video: bool = True,
            audio_transform_method = 'fbank', # Select audio transform method
            audio_sample_rate = None,
            audio_n_mels = None,
            audio_frame_length = None,
            audio_frame_shift = None,
            audio_normalize = False,
            spec_augment: float = 0.,     # probability of process spec-augmentation
            freq_mask_para: int = None,
            freq_mask_num: int = None,
            time_mask_num: int = None,
            noise_rate: float = 0.,     # probability of adding noise
            noise_path: str = None,
            return_path: bool = False,
            ):
        super(AV_Dataset, self).__init__()
        
        self.video_paths = list(video_paths)
        self.audio_paths = list(audio_paths)
        self.korean_transcripts = list(korean_transcripts)
        self.transcripts = list(transcripts)
        self.dataset_size = len(self.audio_paths)

        self.sos_id=sos_id
        self.eos_id=eos_id
        
        self.audio_sample_rate = audio_sample_rate
        self.audio_transform_method = audio_transform_method
        self.normalize = audio_normalize

        if audio_transform_method.lower() == 'fbank':
            self.audio_feature = FilterBank(
                                audio_sample_rate, 
                                audio_n_mels, 
                                audio_frame_length, 
                                audio_frame_shift,
                              )
        elif audio_transform_method.lower() == 'mel':
            self.audio_feature = MelSpectrogram(
                                audio_sample_rate, 
                                audio_n_mels, 
                                audio_frame_length, 
                                audio_frame_shift,
                              )
        self.use_audio = use_audio
        self.use_video = use_video
        self.raw_video = raw_video

        self.return_path = return_path
        
        self.spec_aug_rate = spec_augment
        self.noise_aug_rate = noise_rate
        # self.augment_methods = [self.NOISE_AUGMENT if noise_augment else self.VANILLA] * len(self.audio_paths)
        #self.augment_methods = np.random.choice([0,1,2,3], len(self.audio_paths), p=[0.6, 0.15, 0.15, 0.1])
        
        if spec_augment:
            self.spec_augment = SpecAugment(freq_mask_para, 
                                            time_mask_num, 
                                            freq_mask_num,)
        if noise_rate > 0.:
            self.noise_augment = BabbleNoise(noise_path, audio_sample_rate)

        # self._augment(spec_augment, noise_augment)

    def __getitem__(self, index):
        if self.use_video:
            video_feature = self.parse_video(self.video_paths[index])
        else:
            # return dummy video feature
            video_feature = torch.Tensor([[0]])
        if self.use_audio:
            prob_all = self.spec_aug_rate * self.noise_aug_rate
            aug_method = np.random.choice(
                [self.VANILLA, self.SPEC_AUGMENT, self.NOISE_AUGMENT, self.BOTH_AUGMENT], 
                p=[1-self.spec_aug_rate-self.noise_aug_rate+prob_all, self.spec_aug_rate, self.noise_aug_rate, prob_all])
            audio_feature = self.parse_audio(self.audio_paths[index], augment_method=aug_method)
            # audio_feature = self.parse_audio(self.audio_paths[index],self.augment_methods[index])
        else:
            # return dummy audio feature
            audio_feature = torch.Tensor([[0]])
        transcript = self.parse_transcript(self.transcripts[index])
        korean_transcript = self.parse_korean_transcripts(self.korean_transcripts[index])
        if self.return_path:
            return video_feature, audio_feature, transcript, korean_transcript, self.video_paths[index]
        return video_feature, audio_feature, transcript, korean_transcript,
        
    def audio_transform(self, audio):
        if self.audio_transform_method == 'raw':
            output = np.expand_dims(audio,1)
        else:
            fbanks = self.audio_feature(audio)
            output = np.transpose(fbanks, (1,0))
        return output
    
    def parse_audio(self,audio_path: str, augment_method):
        signal, _ = get_sample(audio_path,resample=self.audio_sample_rate)
        if augment_method in [self.NOISE_AUGMENT, self.BOTH_AUGMENT]:
            signal = self.noise_augment(signal, is_path=False)
        
        feature = _parse_audio(signal, self.audio_transform, self.normalize)
        
        if augment_method in [self.SPEC_AUGMENT, self.BOTH_AUGMENT]:
            feature = self.spec_augment(feature)
            
        return feature
    
    def parse_video(self, video_path: str):
        video = _parse_video(video_path, is_raw=self.raw_video)
        return video

    def parse_transcript(self, transcript):
        transcript = _parse_transcript(transcript, self.sos_id, self.eos_id)
        return transcript
    
    def parse_korean_transcripts(self, korean_transcript):
        korean_transcript = _parse_korean_transcript(korean_transcript, self.sos_id, self.eos_id)
        return korean_transcript

    def _augment(self, spec_augment, noise_augment):
        """ Spec Augmentation """
        available_augment = list()

        if spec_augment:
            available_augment.append(self.SPEC_AUGMENT)
        elif noise_augment:
            available_augment.append(self.NOISE_AUGMENT)
        
        if available_augment:
            print(f"Applying Augmentation...{self.dataset_size}")
            for idx in range(self.dataset_size):
                self.augment_methods.append(np.random.choice(available_augment))
                self.video_paths.append(self.video_paths[idx])
                self.audio_paths.append(self.audio_paths[idx])
                self.korean_transcripts.append(self.korean_transcripts[idx])
                self.transcripts.append(self.transcripts[idx])

            print(f"Augmentation Finished...{len(self.audio_paths)}")

    def shuffle(self):
        """ Shuffle dataset """
        print('shuffle dataset')
        tmp = list(zip(self.video_paths,self.audio_paths,self.korean_transcripts, self.transcripts))
        random.shuffle(tmp)
        self.video_paths,self.audio_paths, self.korean_transcripts, self.transcripts = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)

def get_batch_from_paths(config, vocab, tr_video_paths, tr_audio_paths, tr_transcripts):
    batch = list()

    if config['audio_transform_method'] == 'fbank':
        audio_feature = FilterBank(
                                config["audio_sample_rate"], 
                                config["audio_n_mels"], 
                                config["audio_frame_length"], 
                                config["audio_frame_shift"],
                            )
    elif config['audio_transform_method'] == 'mel':
        audio_feature = MelSpectrogram(
                                config["audio_sample_rate"], 
                                config["audio_n_mels"], 
                                config["audio_frame_length"], 
                                config["audio_frame_shift"],
                            )
    else:
        audio_feature = None

    if config['audio_transform_method'] == 'raw':
        audio_transform = lambda x: np.expand_dims(x,1)
    else:
        audio_transform = lambda x: np.transpose(audio_feature(x), (1,0))

    for tr_video_path, tr_audio_path, tr_transcript in zip(tr_video_paths, tr_audio_paths, tr_transcripts):
        vids = _parse_video(tr_video_path, is_raw=config["raw_video"])
        signal, _ = get_sample(tr_audio_path,resample=config['audio_sample_rate'])
        seqs = _parse_audio(signal, audio_transform, config['audio_normalize'])
        targets = _parse_transcript(tr_transcript, vocab.sos_id, vocab.eos_id)
        batch.append((vids, seqs, targets))

    return batch

class AVcollator:
    def __init__(
        self, 
        max_len : int = None,
        use_video : bool = True,
        raw_video : bool = True,
        infer : bool = False,
    ):
        self.max_len = max_len
        self.use_video = use_video
        self.raw_video = raw_video
        self.infer = infer
        
    def __call__(self, batch):
        if self.infer:
            return _infer_collate_fn(batch, self.max_len, self.use_video, self.raw_video)
        return _collate_fn(batch, self.max_len, self.use_video, self.raw_video)


def _collate_fn(
    batch, 
    max_target_len : int = None,
    use_video : bool = True,
    raw_video : bool = True,
):
    """ functions that pad to the maximum sequence length """
    def vid_length_(p):
        return len(p[0])

    def seq_length_(p):
        return len(p[1])

    def target_length_(p):
        return len(p[2])
    
    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    
    seq_lengths = [len(s[1]) for s in batch]
    target_lengths = [min(max_target_len, len(s[2]))-1 for s in batch]
    target_lengths = torch.IntTensor(target_lengths)
    
    max_seq_sample = max(batch, key=seq_length_)[1]
    # max_target_sample = max(batch, key=target_length_)[2]
    
    max_seq_size = max_seq_sample.size(0)
    # max_target_size = len(max_target_sample)
    max_target_size = max_target_len
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)
    
    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(0)
    
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[1]
        target = sample[2]
        target = target[:max_target_size]
        seq_length = tensor.size(0)
        # in 0 dim, mask 0 to seq_length
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    
    seq_lengths = torch.IntTensor(seq_lengths)
    seqs = seqs.permute(0,2,1) # B T C  --> B C T
    
    if use_video :
        vid_lengths = [vid_length_(s) for s in batch]
        max_vid_batch = max(batch, key=vid_length_)
        max_vid_sample = max_vid_batch[0]
        max_vid_size = vid_length_(max_vid_batch)
            
        if raw_video:
            vid_feat_x = max_vid_sample.size(1)
            vid_feat_y = max_vid_sample.size(2)
            vid_feat_c = max_vid_sample.size(3)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_x,vid_feat_y,vid_feat_c)
        else:
            vid_feat_c = max_vid_sample.size(1)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_c)
        
        for x in range(batch_size):
            sample = batch[x]
            video_ = sample[0]
            
            if raw_video:
                vids[x,:video_.size(0),:,:,:] = video_
            else:
                vids[x,:video_.size(0),:] = video_
    
        vid_lengths = torch.IntTensor(vid_lengths)
        
        if raw_video:
            # B T W H C --> B C T W H
            # pdb.set_trace()
            vids = vids.permute(0,4,1,2,3)

    else:
        vids = torch.zeros((batch_size, 1))
        vid_lengths = torch.zeros((batch_size,)).to(int)
    
    """
    print('show sample size')
    print(f"video_size = {vids.size()}")
    print(f"audio_size = {seqs.size()}")
    """
    
    return vids, seqs, targets, vid_lengths, seq_lengths, target_lengths


def _infer_collate_fn(
    batch, 
    max_target_len : int = None,
    use_video : bool = True,
    raw_video : bool = True,
):
    """ functions that pad to the maximum sequence length """
    def vid_length_(p):
        return len(p[0])

    def seq_length_(p):
        return len(p[1])

    def target_length_(p):
        return len(p[2])
    
    seq_lengths = [len(s[1]) for s in batch]
    target_lengths = [min(max_target_len, len(s[2]))-1 for s in batch]
    target_lengths = torch.IntTensor(target_lengths)
    
    max_seq_sample = max(batch, key=seq_length_)[1]
    # max_target_sample = max(batch, key=target_length_)[2]
    
    max_seq_size = max_seq_sample.size(0)
    # max_target_size = len(max_target_sample)
    max_target_size = max_target_len
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)
    
    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(0)

    paths = list()
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[1]
        target = sample[2]
        target = target[:max_target_size]
        paths.append(sample[-1])
        seq_length = tensor.size(0)
        # in 0 dim, mask 0 to seq_length
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
    
    seq_lengths = torch.IntTensor(seq_lengths)
    seqs = seqs.permute(0,2,1) # B T C  --> B C T
    
    if use_video :
        vid_lengths = [vid_length_(s) for s in batch]
        max_vid_batch = max(batch, key=vid_length_)
        max_vid_sample = max_vid_batch[0]
        max_vid_size = vid_length_(max_vid_batch)
            
        if raw_video:
            vid_feat_x = max_vid_sample.size(1)
            vid_feat_y = max_vid_sample.size(2)
            vid_feat_c = max_vid_sample.size(3)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_x,vid_feat_y,vid_feat_c)
        else:
            vid_feat_c = max_vid_sample.size(1)
            vids = torch.zeros(batch_size, max_vid_size, vid_feat_c)
        
        for x in range(batch_size):
            sample = batch[x]
            video_ = sample[0]
            
            if raw_video:
                vids[x,:video_.size(0),:,:,:] = video_
            else:
                vids[x,:video_.size(0),:] = video_
    
        vid_lengths = torch.IntTensor(vid_lengths)
        
        if raw_video:
            # B T W H C --> B C T W H
            # pdb.set_trace()
            vids = vids.permute(0,4,1,2,3)

    else:
        vids = torch.zeros((batch_size, 1))
        vid_lengths = torch.zeros((batch_size,)).to(int)
    
    """
    print('show sample size')
    print(f"video_size = {vids.size()}")
    print(f"audio_size = {seqs.size()}")
    """
    
    return vids, seqs, targets, vid_lengths, seq_lengths, target_lengths, paths
