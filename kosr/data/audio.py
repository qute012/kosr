import soundfile as sf
import numpy as np
import torch
import torchaudio

def load_audio(path, sr=16000, out_tensor=True):
    ext = path.split('.')[-1]

    if ext=='pcm':
        try:
            sig, sr = np.memmap(path, dtype='h', mode='r').astype('float32'), sr
        except:
            with open (path, 'rb') as f:
                buf = f.read()
                if len(buf)%2==1:
                    buf = buf[:-1]
            sig, sr = np.frombuffer(buf, dtype='int16'), sr
    else:
        sig, sr = sf.read(path, sr)
    
    if out_tensor:
        return torch.FloatTensor(sig), sr
    else:
        return sig, sr