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
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
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
        #mel = torch.log1p(mel)
        spec = self.amplitude_to_db(mel)
        #log_offset = 1e-20
        #mel = torch.log(mel+log_offset)
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
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
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
        #spec = torch.log1p(spec)
        #spec = self.amplitude_to_db(spec)
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
            sample_frequency = self.sample_rate,
            window_type = 'hanning'
        )
        spec = spec.transpose(1,0)
        if self.normalized:
            spec = self.norm(mel)
        return spec
    
class SpecAugment(object):
    def __init__(self, prob=0.5, n_mask=2, F=30, T=40, masking='mean'):
        self.n_mask = n_mask
        self.F = F
        self.T = T
        self.masking = masking
        self.probability = prob

    def __call__(self, spec):
        if random.random() < self.probability:
            nu, tau = spec.shape
            
            for _ in range(self.n_mask):
                f = random.randint(0, self.F)
                f0 = random.randint(0, nu - f)
                if self.masking=='zero':
                    spec[f0:f0 + f, :] = 0
                else:
                    spec[f0:f0 + f, :] = spec[f0:f0 + f, :].mean()


                t = random.randint(0, self.T)
                t0 = random.randint(0, tau - t)
                if self.masking=='zero':
                    spec[:, t0:t0 + t] = 0
                else:
                    spec[:, t0:t0 + t] = spec[:, t0:t0 + t].mean()

        return spec