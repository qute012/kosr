import warnings
import yaml
warnings.filterwarnings('ignore')

from kosr.model import Transformer
from kosr.trainer import train, valid
from kosr.utils.loss import LabelSmoothingLoss
from kosr.utils.optimizer import get_std_opt
from kosr.data.dataset import get_dataloader
from kosr.utils.convert import vocab

def main():
    conf = 'config/ksponspeech.yaml'
    with open(conf, 'r') as f:
        conf = yaml.safe_load(f)
        
    batch_size = 4
    train_dataloader = get_dataloader('data/Ksponspeech/train.trn', batch_size=batch_size)
    valid_dataloader = get_dataloader('data/Ksponspeech/dev.trn', batch_size=batch_size)
    test_dataloader = get_dataloader('data/Ksponspeech/eval_clean.trn', batch_size=batch_size)
    
    model = Transformer(len(vocab)).cuda()
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=0, smoothing=0.1).cuda()
    optimizer = get_std_opt(model.parameters(), **conf['optimizer'])
    
    for epoch in range(30):
        train(model, optimizer, criterion, train_dataloader, epoch)
        valid(model, optimizer, criterion, valid_dataloader, epoch)

if __name__ == '__main__':
    main()