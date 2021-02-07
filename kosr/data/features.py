import torch
import torchaudio
import librosa
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class MelSpectrogram(object):
    def __init__(self, sample_rate=16000, win_length=0.02, win_stride=0.01, n_mels=80, normalized=False, library='torchaudio'):
        self.sample_rate = sample_rate
        self.hop_length = int(self.sample_rate * win_stride)
        self.n_fft = int(self.sample_rate * win_length)
        self.win_length = self.n_fft
        self.n_mels = n_mels
        self.normalized = normalized
        
        #self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.library = library
        
        if self.library=='librosa':
            self.transform = librosa.feature.melspectrogram
        else:
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=n_mels,
            )
        
    def norm(self, spec, eps=1e-5):
        m = torch.mean(spec, axis=1, keepdims=True)
        s = torch.std(spec, axis=1, keepdims=True) + eps
        return (spec-m)/s

    def __call__(self, signal):
        if self.library=='librosa':
            mel = self.transform(
                signal, 
                sr=self.sample_rate, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
        else:
            mel = self.transform(signal)
        
        #mel = self.amplitude_to_db(mel)
        mel = torch.log1p(mel)
        if self.normalized:
            mel = self.norm(mel)
        return mel

class Spectrogram(object):
    def __init__(self, sample_rate=16000, win_length=0.02, win_stride=0.01, normalized=False):
        self.sample_rate = sample_rate
        self.hop_length = int(self.sample_rate * win_stride)
        self.n_fft = int(self.sample_rate * win_length)
        self.win_length = self.n_fft
        self.normalized = normalized
        
        #self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length=self.hop_length,
        )
        
    def norm(self, spec, eps=1e-5):
        m = torch.mean(spec, axis=1, keepdims=True)
        s = torch.std(spec, axis=1, keepdims=True) + eps
        return (spec-m)/s

    def __call__(self, signal):
        spec = self.transform(signal)
        spec = torch.log1p(spec)
        #mel = self.amplitude_to_db(mel)
        if self.normalized:
            spec = self.norm(mel)
        return spec

class FBank(object):
    def __init__(self, sample_rate=16000, win_length=0.02, win_stride=0.01, n_mels=80, normalized=False):
        self.sample_rate = sample_rate
        self.frame_length = win_length * 1000
        self.frame_shift = win_stride * 1000
        self.n_mels = n_mels
        self.normalized = normalized
        
        #self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.transform = torchaudio.compliance.kaldi.fbank
        
    def norm(self, spec, eps=1e-5):
        m = torch.mean(spec, axis=1, keepdims=True)
        s = torch.std(spec, axis=1, keepdims=True) + eps
        return (spec-m)/s

    def __call__(self, signal):
        spec = self.transform(
            signal,
            num_mel_bins = self.n_mels,
            frame_length = self.frame_length,
            frame_shift = self.frame_shift,
            sample_frequency = self.sample_rate
        )
        spec = spec.transpose(1,0)
        if self.normalized:
            spec = self.norm(mel)
        return spec
    
class SpecAugment(object):
    def __init__(self, frequency_mask_max_percentage=0.3, time_mask_max_percentage=0.1, prob=0.5):
        self.frequency_mask_probability = frequency_mask_max_percentage
        self.time_mask_probability = time_mask_max_percentage
        self.probability = prob

    def __call__(self, spec):
        if random.random() < self.probability:
            nu, tau = spec.shape

            f = random.randint(0, int(self.frequency_mask_probability*nu))
            f0 = random.randint(0, nu - f)
            spec[f0:f0 + f, :] = 0

            t = random.randint(0, int(self.time_mask_probability*tau))
            t0 = random.randint(0, tau - t)
            spec[:, t0:t0 + t] = 0

        return spec