import datetime
import os
import yaml
import logging

def build_conf(conf_path='config/ksponspeech.yaml'):
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)
    from kosr.utils.convert import vocab
    if not 'n_mels' in conf['feature']['spec'].keys():
        conf['model']['in_dim'] = int(conf['feature']['spec']['sample_rate'] 
                                      * conf['feature']['spec']['win_length']/2) + 1
    conf['model']['out_dim'] = len(vocab)
    
    return conf

def get_now():
    now = datetime.datetime.now()
    cur = now.strftime('%m-%d-%H:%M')
    
    return cur

def make_chk(root='checkpoint'):
    cur = get_now()
    path = os.path.join(root,cur)
    
    return path, cur

chk_path, cur = make_chk()

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('log/{}.log'.format(cur))

logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(level=logging.DEBUG)

train_log = "[{}] epoch: {} loss: {:.2f} cer: {:.2f} lr: {:.7f}"
valid_log = "[{}] epoch: {} cer: {:.2f} wer: {:.2f}"
eval_log = "[{}] cer: {:.2f} wer: {:.2f}"
epoch_log = "[{}] {} epoch is over. {} epoch best wer: {:.2f} {} epoch best loss: {:.2f}"