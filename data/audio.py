import soundfile as sf
import numpy as np
import torch
import torchaudio

def load_audio(path, sr=16000):
    ext = path.split('.')[-1]
    
    if ext=='pcm':
        with open (path, 'rb') as f:
            buf = f.read()
            if len(buf)%2==1:
                buf = buf[:-1]
            sig = np.frombuffer(buf, dtype='int16')
    else:
        sig, sr = sf.read(path, sr)
            
    sig = torch.FloatTensor(sig).unsqueeze(0)
    return sig, sr