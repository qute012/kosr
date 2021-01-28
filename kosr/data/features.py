import torch
import torchaudio
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class MelSpectrogram(object):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=256, n_mels=80, normalized=True):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalized = normalized
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
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
        mel = self.transform(signal)
        mel = self.amplitude_to_db(mel)
        if self.normalized:
            mel = self.norm(mel)
        return mel

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