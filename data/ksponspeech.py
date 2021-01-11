import json
import yaml
import argparse
import os
from tqdm import tqdm

from features import *
from audio import *

import pickle

def prep_manifest(trn, root_dir):
    with open('Ksponspeech/kspon_labels.json', 'r') as f:
        vocab = json.load(f)
    
    os.makedirs('Ksponspeech/features', exist_ok=True)
    with open(trn, 'r') as f:
        lines = f.read().strip().split('\n')
    
    with open('../config/ksponspeech.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    transforms = Compose([
        MelSpectrogram(**conf['feature']['spec'])
    ])
    
    feat_path = os.path.join('Ksponspeech/features',trn.split('/')[-1].split('.')[0])
    os.makedirs(feat_path, exist_ok=True)
    
    record_size = 10000
    cur_rec = 0
    cnt_spec = 0
    records = list()
    for line in tqdm(lines):
        fname, script = line.split(' :: ')
        fname = os.path.join(root_dir, fname)
        sig, sr = load_audio(fname)
        spec = transforms(sig)
        
        records.append((spec,script))
        cnt_spec += 1
        if cnt_spec==record_size:
            with open(os.path.join(feat_path,f"{cur_rec}.record"),'wb') as f:
                pickle.dump(records, f)
            cnt_spec = 0
            cur_rec += 1
            records = list()
    if len(records)>0:
        with open(os.path.join(feat_path,f"{cur_rec}.record"),'wb') as f:
            pickle.dump(records, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparing Ksponspeech for training and evaluating')
    parser.add_argument('--path', default='/root/storage/dataset/kspon', type=str, help="dataset root directory")
    parser.add_argument('--workers', default=4, type=int, help="number of workers")
    args = parser.parse_args()
    
    name = "Ksponspeech"
    
    root_dir = args.path
    for trn in ['train.trn', 'dev.trn', 'eval_clean.trn', 'eval_other.trn']:
        prep_manifest(os.path.join(name, trn), root_dir)