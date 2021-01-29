import datetime
import os

def get_now():
    now = datetime.datetime.now()
    cur = now.strftime('%m-%d %H:%M')
    
    return cur

def make_chk(root='checkpoint'):
    path = os.path.join(root,get_now())
    os.makedirs(dirs, exist_ok=True)
    
    return path

train_log = "[{}] epoch: {} loss: {:.2f} cer: {:.2f} lr: {:.7f}"
valid_log = "[{}] epoch: {} loss: {:.2f} cer: {:.2f} wer: {:.2f}"
epoch_log = "[{}] {} epoch is over. {} epoch best wer: {} {} epoch best loss: {}"