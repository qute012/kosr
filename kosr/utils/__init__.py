import datetime
import os

def get_now():
    now = datetime.datetime.now()
    cur = now.strftime('%m-%d %H:%M')
    return cur

def make_chk(epoch, model_type='tf', root='checkpoint'):
    path = os.path.join(root,get_now())
    dirs = '/'.join(path.split('/')[:-1])
    os.makedirs(dirs, exist_ok=True)
    
    return os.path.join(path, f"{epoch}_{model_type}.pth"