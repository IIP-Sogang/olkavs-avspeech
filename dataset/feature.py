import torch
import platform
import numpy as np
from torch import Tensor, FloatTensor
import torchaudio

if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile")
elif platform.system() == 'Linux':
    torchaudio.set_audio_backend("sox_io")


class Spectrogram(object):
    """
    Create a spectrogram from a audio signal.
    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction (default: torch)
    """
    def __init__(
            self,
            sample_rate: int = 16000,
            frame_length: int = 40,
            frame_shift: int = 10,
            feature_extract_by: str = 'torch') :
        self.sample_rate = sample_rate
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'kaldi':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            try:
                import torchaudio
            except ImportError:
                raise ImportError("Please install torchaudio: `pip install torchaudio`")

            self.transforms = torchaudio.compliance.kaldi.spectrogram
            self.frame_length = frame_length
            self.frame_shift = frame_shift

        else:
            self.n_fft = int(round(sample_rate * 0.001 * frame_length))
            self.hop_length = int(round(sample_rate * 0.001 * frame_shift))

    def __call__(self, signal):
        if self.feature_extract_by == 'kaldi':
            spectrogram = self.transforms(
                Tensor(signal).unsqueeze(0),
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                sample_frequency=self.sample_rate,
            ).transpose(0, 1)

        else:
            spectrogram = torch.stft(
                Tensor(signal), self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=torch.hamming_window(self.n_fft),
                center=False, normalized=False, onesided=True
            )
            spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
            spectrogram = np.log1p(spectrogram.numpy())

        return spectrogram


class MelSpectrogram(object):

    def __init__(
            self,
            sample_rate: int = 16000,
            n_mels: int = 80,
            frame_length: int = 40,
            frame_shift: int = 10,
            feature_extract_by: str = 'torchaudio'
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'torchaudio':
            # torchaudio is only supported on Linux (Linux, Mac)
            self.transforms = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                win_length=frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=n_mels,
            )

    def __call__(self, signal):
        if self.feature_extract_by == 'torchaudio':
            melspectrogram = self.transforms(signal)
            melspectrogram = melspectrogram.squeeze(0)
        return melspectrogram # F T


class MFCC(object):
    """
    Create the Mel-frequency cepstrum coefficients (MFCCs) from an audio signal.
    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mfcc (int):  Number of mfc coefficients to retain. (Default: 40)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
        feature_extract_by (str): which library to use for feature extraction(default: librosa)
    """
    def __init__(
            self,
            sample_rate: int = 16000,
            n_mfcc: int = 40,
            frame_length: int = 20,
            frame_shift: int = 10,
            feature_extract_by: str = 'librosa'
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = int(round(sample_rate * 0.001 * frame_length))
        self.hop_length = int(round(sample_rate * 0.001 * frame_shift))
        self.feature_extract_by = feature_extract_by.lower()

        if self.feature_extract_by == 'torchaudio':
            # torchaudio is only supported on Linux (Linux, Mac)
            assert platform.system().lower() == 'linux' or platform.system().lower() == 'darwin'
            import torchaudio

            self.transforms = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                log_mels=True,
                win_length=frame_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
            )
        else:
            import librosa
            self.transforms = librosa.feature.mfcc

    def __call__(self, signal):
        if self.feature_extract_by == 'torchaudio':
            mfcc = self.transforms(FloatTensor(signal))
            mfcc = mfcc.numpy()

        elif self.feature_extract_by == 'librosa':
            mfcc = self.transforms(
                y=signal,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

        else:
            raise ValueError("Unsupported library : {0}".format(self.feature_extract_by))

        return mfcc


class FilterBank(object):
    '''
    Create a fbank from a raw audio signal. This matches the input/output of Kaldi’s compute-fbank-feats
    Args:
        sample_rate (int): Sample rate of audio signal. (Default: 16000)
        n_mels (int):  Number of mfc coefficients to retain. (Default: 80)
        frame_length (int): frame length for spectrogram (ms) (Default : 20)
        frame_shift (int): Length of hop between STFT windows. (ms) (Default: 10)
    '''
    def __init__(
            self,
            sample_rate: int = 16000,
            n_mels: int = 80,
            frame_length: int = 20,
            frame_shift: int = 10
    ) -> None:
        try:
            import torchaudio
        except ImportError:
            raise ImportError("Please install torchaudio `pip install torchaudio`")
        self.transforms = torchaudio.compliance.kaldi.fbank
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

    def __call__(self, signal):
        return self.transforms(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        ).transpose(0, 1).numpy()