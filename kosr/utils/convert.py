import json

PAD_TOKEN = '<pad>'
UNK_TOKEN = None
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

with open('data/Ksponspeech/kspon_labels.json', 'r') as f:
        vocab = json.load(f)
        
id2char = list(vocab)
char2id = dict()

for i, c in enumerate(id2char):
    char2id[c] = i
    