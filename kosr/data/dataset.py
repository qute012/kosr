import os
import yaml
from torch.utils.data import Dataset, DataLoader

from kosr.data.features import *
from kosr.data.audio import *
from kosr.utils.convert import char2id, id2char

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

class SpeechDataset(Dataset):
    def __init__(self, trn, root_dir='/root/storage/dataset/kspon', mode='train', conf='config/ksponspeech.yaml'):
        super(SpeechDataset, self).__init__()
        self.root_dir = root_dir
        with open(trn, 'r') as f:
            self.data = f.read().strip().split('\n')
        self.prep_data()
        
        with open(conf, 'r') as f:
            self.conf = yaml.safe_load(f)
            
        self.transforms = Compose([
            MelSpectrogram(**self.conf['feature']['spec']),
            SpecAugment(prob=self.conf['feature']['augment']['spec_augment'])
        ])
        
    def prep_data(self, symbol=' :: '):
        """ 
        Data preparations.
        (A, B): Tuple => A: Audio file path, B: Transcript
        """
        temp = []
        for line in self.data:
            fname, script = line.split(' :: ')
            fname = os.path.join(self.root_dir, fname)
            temp.append((fname,script))
        self.data = temp
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        fname, script = self.data[index]
        sig, sr = load_audio(fname)
        spec = self.transforms(sig).transpose(1,0)
        seq = self.scr_to_seq(script)
        return spec, seq
        
    def scr_to_seq(self, scr):
        seq = list()
        seq.append(char2id.get(SOS_TOKEN))
        for c in scr:
            if c in id2char:
                seq.append(char2id.get(c))
            else:
                if UNK_TOKEN is not None:
                    seq.append(char2id.get(unk))
                else:
                    continue
        seq.append(char2id.get(EOS_TOKEN))
        return seq
        


class FileDataset(Dataset):
    def __init__(self, root='Ksponspeech/features'):
        super(FileDataset, self).__init__()
        raise NotImplementedError
         
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
        
def get_dataloader(trn, root_dir='/root/storage/dataset/kspon', batch_size=16, mode='train'):
    shuffle = True if mode=='train' else False
    return DataLoader(SpeechDataset(trn, root_dir), batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                              collate_fn=_collate_fn, num_workers=8)
        
def _collate_fn(batch):
    """ functions that pad to the maximum sequence length """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths