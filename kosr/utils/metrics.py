import Levenshtein as Lev
from kosr.data.dataset import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from kosr.utils.convert import char2id, id2char

def metrics(preds, targets):
    btz = targets.size(0)
    cers = 0.
    wers = 0.
    preds_str = seq_to_str(preds, id2char)
    golds_str = seq_to_str(targets, id2char)
    for i, (pred,gold) in enumerate(zip(preds_str,golds_str)):
        try:
            length = len(gold.replace(' ',''))
            cers += cer(pred,gold)/length
            length = len(gold.split())
            wers += wer(pred,gold)/length
        except:
            print(pred, gold)
            assert 1==0
    return cers/btz, wers/btz

def wer(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Lev.distance(''.join(w1), ''.join(w2))


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def seq_to_str(seqs, id2char):
    assert len(seqs.shape)<=2, 'can not convert 3-dimensional sequence to string'
    pad_id = id2char.index(PAD_TOKEN)
    unk_id = id2char.index(UNK_TOKEN)
    sos_id = id2char.index(SOS_TOKEN)
    eos_id = id2char.index(EOS_TOKEN)
    
    if len(seqs.shape) == 1:
        sentence = str()
        for idx in seqs:
            idx = idx.item()
            if idx==sos_id or idx==pad_id or idx==unk_id:
                continue
            if idx==eos_id:
                break
            sentence += id2char[idx]
        return sentence

    elif len(seqs.shape) == 2:
        sentences = list()
        for seq in seqs:
            sentence = str()
            for idx in seq:
                idx = idx.item()
                if idx==sos_id or idx==pad_id or idx==unk_id:
                    continue
                if idx==eos_id:
                    break
                sentence += id2char[idx]
            sentences.append(sentence)
        return sentences