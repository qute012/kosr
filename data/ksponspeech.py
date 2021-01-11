import json
import yaml
import argparse

parser = argparse.ArgumentParser(description='Preparing Ksponspeech for training and evaluating')
parser.add_argument('--path', default='/root/storage/dataset/kspon', type=str, help="dataset root directory")
args = parser.parse_args()

root_dir = args.path
    
with open('Ksponspeech/kspon_labels.json', 'r') as f:
    vocab = json.load(f)
    
with open('../config/ksponspeech.yaml', 'r') as f:
    conf = yaml.load(f)

