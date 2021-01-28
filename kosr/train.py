import warnings
import yaml
warnings.filterwarnings('ignore')

from kosr.model import Transformer
from kosr.trainer import train
from kosr.utils.loss import LabelSmoothingLoss
from kosr.utils.optimizer import get_std_opt
from kosr.data.dataset import get_dataloader
from kosr.utils.convert import vocab

def main():
    conf = 'config/ksponspeech.yaml'
    with open(conf, 'r') as f:
        conf = yaml.safe_load(f)
        
    train_dataloader = get_dataloader('data/Ksponspeech/train.trn')
    valid_dataloader = get_dataloader('data/Ksponspeech/dev.trn')
    test_dataloader = get_dataloader('data/Ksponspeech/eval_clean.trn')
    
    model = Transformer(len(vocab)).cuda()
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=0, smoothing=0.1).cuda()
    optimizer = get_std_opt(model.parameters(), **conf['optimizer'])
    
    train(model, optimizer, criterion, train_dataloader, 0)

if __name__ == '__main__':
    main()