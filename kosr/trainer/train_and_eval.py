import torch
import torch.nn as nn
from tqdm import tqdm

from kosr.utils.metrics import metrics

import logging
logging.basicConfig(filename='log/train.log',level=logging.INFO)

train_log = "[{}] epoch: {} loss: {} cer: {} last batch cer: {}"
valid_log = "[{}] epoch: {} loss: {} cer: {} wer: {}"

def train(model, optimizer, criterion, dataloader, epoch, max_norm=400, print_step=100):
    losses = 0.
    cer_sum = 0.
    wer_sum = 0.
    model.train()
    pbar = tqdm(dataloader)
    for batch in pbar:
        optimizer.zero_grad()

        inputs, targets, input_length, target_length = batch
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
            input_length = input_length.cuda()
            target_length = target_length.cuda()
        
        preds = model(inputs. input_length, targets)
        
        loss = criterion(preds.view(-1, preds.size(-1)), targets[:,1:].view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        
        losses += loss.item()
        
        y_hats = preds.max(-1)[1]
        
        _cer, _wer = metrics(y_hats, targets)
        cer += _cer
        wer += _wer
        step += 1
        pbar.set_description(log_info.format(epoch, losses/step, cer/step, _cer))
        if step%print_step==0:
            logging.info(train_log.format('training', epoch, losses/step, cer/step, _cer))
        
def valid(model, optimizer, criterion, dataloader, epoch):
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

            preds, y_hats = greedy_search(inputs. input_length)

            loss = criterion(preds.view(-1, preds.size(-1)), targets[:,1:].view(-1))

            losses += loss.item()

            _cer, _wer = metrics(y_hats, targets)
            cer += _cer
            wer += _wer
            step += 1
    logging.info(valid_log.format('valid', epoch, losses/step, cer/step, wer/step))