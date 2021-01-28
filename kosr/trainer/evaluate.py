import torch
import torch.nn as nn
from tqdm import tqdm

from kosr.utils.metrics import metrics

import logging
logging.basicConfig(filename='log/train.log',level=logging.INFO)

eval_log = "[{}] loss: {} cer: {} wer: {}"

def valid(model, optimizer, criterion, dataloader):
    losses = 0.
    cer_sum = 0.
    wer_sum = 0.
    model.eval()
    pbar = tqdm(dataloader)
    with torch.no_grad():
        for batch in pbar:
            inputs, targets, input_length, target_length = batch

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                input_length = input_length.cuda()
                target_length = target_length.cuda()

            preds, y_hats = model.recognize(inputs. input_length)

            loss = criterion(preds.view(-1, preds.size(-1)), targets[:,1:].view(-1))

            losses += loss.item()

            _cer, _wer = metrics(y_hats, targets)
            cer += _cer
            wer += _wer
            step += 1
    logging.info(eval_log.format('evaluate', losses/step, cer/step, wer/step))