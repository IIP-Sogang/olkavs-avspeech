import os, pdb
import random
import math
from typing import List

import numpy as np
import tqdm
import torchaudio
from torchaudio import functional as F
import torch
from torch import Tensor

import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile")
elif platform.system() == 'Linux':
    torchaudio.set_audio_backend("sox_io")

import sys
INT_INF = sys.maxsize # 2^63 - 1

class SpecAugment(object):
    def __init__(self, freq_mask_para: int = 18, time_mask_num: int = 10, freq_mask_num: int = 2) :
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, feature: Tensor) :
        """ Provides SpecAugmentation for audio """
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20      # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0: t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0: f0 + f] = 0

        return feature

def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects, )

def get_sample(path, resample=None):
    sample = _get_sample(path, resample=resample)
    return sample


class BabbleNoise(object):
    def __init__(self, noise_path:str, sr:int, snr_range:List[int]=[-10,0,10,INT_INF]):
        self.sr = sr
        self.snr_range = snr_range
        self.noise_path = noise_path if noise_path[-1] == '/' else noise_path + '/'
        self.noises = []
        print("Load Background Noise Data")
        for file in tqdm.tqdm(os.listdir(noise_path)):
            # bg = cal_amp_from_pth(self.noise_path+file)
            if '.wav' in file:
                bg, _ = get_sample(self.noise_path+file, sr)
                self.noises.append(bg)
        print("Complete!")
    
    def __call__(self,audio,is_path=True):
        snr_db = np.random.choice(self.snr_range)
        if snr_db == INT_INF:
            return audio
        
        noise_index = np.random.randint(len(self.noises))
        noise = self.noises[noise_index]
        if is_path:
            audio, _ = get_sample(audio,self.sr)

        assert audio.size(0)==1, "Multi-channel data is unavailable!!"
        output = synthesize(audio[0], noise[0], snr_db)
        output = output.unsqueeze(0)
        return output

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a) 
    return noise_rms

def cal_amp_torchaudio(path, new_sr=None):
    audio, sr = torchaudio.load(path)
    if new_sr:
        audio = torchaudio.transforms.Resample(sr, new_sr)(audio)
    audio = audio.permute(1,0).reshape(-1)*32768
    return audio.numpy(), sr

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def synthesize(clean_amp, noise_amp, snr):
    clean_amp = clean_amp.clone().numpy()
    noise_amp = noise_amp.clone().numpy()

    clean_amp *= 32768
    noise_amp *= 32768

    clean_rms = cal_rms(clean_amp)

    start = random.randint(0, len(noise_amp)-len(clean_amp))
    divided_noise_amp = noise_amp[start: start + len(clean_amp)]
    noise_rms = cal_rms(divided_noise_amp)

    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    
    adjusted_noise_amp = divided_noise_amp * (adjusted_noise_rms / noise_rms) 
    mixed_amp = (clean_amp + adjusted_noise_amp)

    #Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed_amp.max(axis=0) > max_int16 or mixed_amp.min(axis=0) < min_int16:
        if mixed_amp.max(axis=0) >= abs(mixed_amp.min(axis=0)): 
            reduction_rate = max_int16 / mixed_amp.max(axis=0)
        else :
            reduction_rate = min_int16 / mixed_amp.min(axis=0)
        mixed_amp = mixed_amp * (reduction_rate)
    
    mixed_amp /= 32768

    mixed_amp = torch.tensor(mixed_amp)

    return mixed_amp
