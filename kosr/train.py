import torch
import warnings
import yaml
import argparse
warnings.filterwarnings('ignore')

from kosr.model import build_model
from kosr.utils import build_conf
from kosr.trainer import train_and_eval, load
from kosr.utils.loss import build_criterion
from kosr.utils.optimizer import build_optimizer
from kosr.data.dataset import get_dataloader
from kosr.utils.convert import vocab

def main(args):
    conf = build_conf(args.conf)
    
    batch_size = conf['train']['batch_size']
    
    train_dataloader = get_dataloader(conf['dataset']['train'], batch_size=batch_size, mode='train', conf=conf)
    valid_dataloader = get_dataloader(conf['dataset']['valid'], batch_size=batch_size, conf=conf)
    test_dataloader = get_dataloader(conf['dataset']['test'], batch_size=batch_size, conf=conf)
    
    model = build_model(conf)
    criterion = build_criterion(conf)
    optimizer = build_optimizer(model.parameters(), **conf['optimizer'])
    
    saved_epoch = load(args, model, optimizer)
    
    train_and_eval(conf['train']['epochs'], model, optimizer, criterion, train_dataloader, valid_dataloader, epoch_save=True, saved_epoch=saved_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Speech Recognition Training')
    parser.add_argument('--conf', default='config/ksponspeech_transformer_base.yaml', type=str, help="configuration path for training")
    parser.add_argument('--load_model', default='', type=str, help="continue to train from saved model")
    args = parser.parse_args()
    main(args)