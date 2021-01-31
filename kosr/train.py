import torch.nn as nn
import warnings
import yaml
warnings.filterwarnings('ignore')

from kosr.model import build_model
from kosr.utils import build_conf
from kosr.trainer import train_and_eval
from kosr.utils.loss import LabelSmoothingLoss
from kosr.utils.optimizer import get_std_opt
from kosr.data.dataset import get_dataloader
from kosr.utils.convert import vocab

def main():
    conf = build_conf('config/ksponspeech.yaml')
        
    batch_size = conf['train']['batch_size']
    train_dataloader = get_dataloader(conf['dataset']['train'], batch_size=batch_size, mode='train')
    valid_dataloader = get_dataloader(conf['dataset']['valid'], batch_size=batch_size)
    test_dataloader = get_dataloader(conf['dataset']['test'], batch_size=batch_size)
    model = build_model(conf)
    criterion = LabelSmoothingLoss(conf['model']['out_dim'], padding_idx=conf['model']['pad_id'], smoothing=0.1).cuda()
    optimizer = get_std_opt(model.parameters(), **conf['optimizer'])
    
    train_and_eval(conf['train']['epochs'], model, optimizer, criterion, train_dataloader, valid_dataloader, epoch_save=True)

if __name__ == '__main__':
    main()