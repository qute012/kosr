import torch
import torch.nn as nn
from tqdm import tqdm

from kosr.utils.metric import metrics

def train(model, optimizer, criterion, dataloader, epoch, max_norm=400):
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
        pbar.set_description("epoch: {} loss: {} cer: {} last batch cer: {}".format(epoch, losses/step, cer/step, _cer))