import soundfile as sf
import torch
import torchaudio

def load_audio(path, sr=16000):
    ext = path.split('.')[-1]
    
    if ext=='pcm':
        sig = np.memmap(path, dtype='h', mode='r')
        sig = sig.astype('float32') / 32767
    else:
        sig, sr = sf.load(path, sr)
    sig = sig.T
    if len(sig.shape) > 1:
        if sig.shape[1] == 1:
            sig = sig.squeeze()
        else:
            sig = sig.mean(axis=1)
    return sig