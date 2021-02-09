import torch
import torch.nn as nn
from tqdm import tqdm

from kosr.utils import eval_log, logger
from kosr.utils.metrics import metrics

def evaluate(model, dataloader):
    losses = 0.
    cer = 0.
    wer = 0.
    step = 0
    model.eval()
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for batch in pbar:
            inputs, targets, input_length, target_length = batch

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            
            preds, y_hats = model.recognize(inputs, input_length, targets)

            _cer, _wer = metrics(y_hats, targets)
            cer += _cer
            wer += _wer
            step += 1
            pbar.set_description(valid_log.format('evaluate', cer/step, wer/step))
    logger.info(valid_log.format('evaluate', cer/step, wer/step))
    
    return cer/step, wer/step