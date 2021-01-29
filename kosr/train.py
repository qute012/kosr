import torch.nn as nn
import warnings
import yaml
warnings.filterwarnings('ignore')

from kosr.model import Transformer
from kosr.trainer import train_and_eval
from kosr.utils.loss import LabelSmoothingLoss
from kosr.utils.optimizer import get_std_opt
from kosr.data.dataset import get_dataloader
from kosr.utils.convert import vocab

def main():
    conf = 'config/ksponspeech.yaml'
    with open(conf, 'r') as f:
        conf = yaml.safe_load(f)
        
    batch_size = conf['train']['batch_size']
    train_dataloader = get_dataloader('data/Ksponspeech/train.trn', batch_size=batch_size, mode='train')
    valid_dataloader = get_dataloader('data/Ksponspeech/dev.trn', batch_size=batch_size)
    test_dataloader = get_dataloader('data/Ksponspeech/eval_clean.trn', batch_size=batch_size)
    model = Transformer(out_dim=len(vocab), **conf['model']).cuda()
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=conf['model']['pad_id'], smoothing=0.1).cuda()
    optimizer = get_std_opt(model.parameters(), **conf['optimizer'])
    
    train_and_eval(conf['train']['epochs'], model, optimizer, criterion, train_dataloader, valid_dataloader, epoch_save=True)

if __name__ == '__main__':
    main()