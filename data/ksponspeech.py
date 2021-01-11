import json
import yaml
import argparse
import os
from tqdm import tqdm

from features import *
from audio import *

def prep_manifest(trn, root_dir):
    with open('Ksponspeech/kspon_labels.json', 'r') as f:
        vocab = json.load(f)

    with open('../config/ksponspeech.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    os.makedirs('Ksponspeech/features', exist_ok=True)
    with open(trn, 'r') as f:
        lines = f.read().strip().split('\n')
        
    for line in tqdm(lines):
        fname, script = line.split(' :: ')
        #print(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preparing Ksponspeech for training and evaluating')
    parser.add_argument('--path', default='/root/storage/dataset/kspon', type=str, help="dataset root directory")
    args = parser.parse_args()
    
    name = "Ksponspeech"
    
    root_dir = args.path
    for trn in ['train.trn', 'dev.trn', 'eval_clean.trn', 'eval_other.trn']:
        prep_manifest(os.path.join(name, trn), root_dir)