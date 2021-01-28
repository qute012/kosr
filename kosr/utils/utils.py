import json

with open('Ksponspeech/kspon_labels.json', 'r') as f:
        vocab = json.load(f)
        
char2id = list(vocab)
id2char = dict()

for i, c in enumerate(char2id):
    id2char[i] = c
    