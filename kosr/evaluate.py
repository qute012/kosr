import torch
import warnings
import yaml
import argparse
warnings.filterwarnings('ignore')

from kosr.model import build_model
from kosr.utils import build_conf
from kosr.trainer import evaluate, load
from kosr.utils.optimizer import build_optimizer
from kosr.data.dataset import get_dataloader

def main(args):
    conf = build_conf(args.conf)
    
    batch_size = conf['train']['batch_size']
    
    test_dataloader = get_dataloader(conf['dataset']['test'], batch_size=batch_size, conf=conf)
    
    model = build_model(conf)
    optimizer = build_optimizer(model.parameters(), **conf['optimizer'])
    
    saved_epoch = load(args, model, optimizer)
    
    evaluate(model, test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Speech Recognition Training')
    parser.add_argument('--conf', default='config/ksponspeech_transformer_base.yaml', type=str, help="configuration path for training")
    parser.add_argument('--load_model', default='', type=str, help="evaluate from saved model")
    args = parser.parse_args()
    main(args)