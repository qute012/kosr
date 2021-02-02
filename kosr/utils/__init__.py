import datetime
import os
import yaml

def build_conf(conf_path='config/ksponspeech.yaml'):
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    from kosr.utils.convert import vocab
    if isinstance(conf['feature']['spec']['win_length'], float):
        conf['model']['in_dim'] = int(conf['feature']['spec']['sample_rate'] 
                                      * conf['feature']['spec']['win_length'])
    conf['model']['out_dim'] = len(vocab)
    
    return conf

def get_now():
    now = datetime.datetime.now()
    cur = now.strftime('%m-%d-%H:%M')
    
    return cur

def make_chk(root='checkpoint'):
    path = os.path.join(root,get_now())
    os.makedirs(path, exist_ok=True)
    
    return path

train_log = "[{}] epoch: {} loss: {:.2f} cer: {:.2f} lr: {:.7f}"
valid_log = "[{}] epoch: {} loss: {:.2f} cer: {:.2f} wer: {:.2f}"
epoch_log = "[{}] {} epoch is over. {} epoch best wer: {} {} epoch best loss: {}"