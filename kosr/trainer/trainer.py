import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, optimizer, criterion, dataloader, epoch, max_norm=150):
    losses = 0.
    cer = 0.
    #wer = 0.
    n_samples = 0
    model.train()
    pbar = tqdm(dataloader)
    for batch in pbar:
        pbar.set_description("epoch: {} loss: {} cer: {}".format(epoch, loss, cer))
        optimizer.zero_grad()

        inputs, targets, input_length, target_length = batch
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
            input_length = input_length.cuda()
            target_length = target_length.cuda()
            
        n_samples += inputs.size(0)
        
        target = y[:, 1:].contiguous().cuda()
        teacher_forcing_rate = scheduler_sampling(epoch)
        logits = model(x, ground_truth=y, teacher_forcing_rate=teacher_forcing_rate)

        y_hats = torch.max(logits, dim=-1)[1]
        #print(label_to_string(target, id2char))
        loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))

        total_loss += loss.item()
        num_samples += batch_size

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()
        cer_, wer_ = score(y_hats.long(), target)
        cer += cer_
        wer += wer_
        if step%print_step==0:
            print('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, wer: {:.2f}, tf_rate: {:.2f}'.format(
                step, total_step, total_loss/num_samples, cer/num_samples, wer/num_samples, teacher_forcing_rate))
            with open('aihub-4.log', 'at') as f:
                f.write('timestep: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, wer: {:.2f}, tf_rate: {:.2f}\n'.format(
                step, total_step, total_loss/num_samples, cer/num_samples, wer/num_samples, teacher_forcing_rate))
        step += 1

    total_loss /= num_samples
    cer /= num_samples
    wer /= num_samples
    print('Epoch %d (Training) Total Loss %0.4f CER %0.4f WER %0.4f' % (epoch, total_loss, cer, wer))
    with open('aihub-4.log', 'at') as f:
        f.write('Epoch %d (Training) Total Loss %0.4f CER %0.4f WER %0.4f\n' % (epoch, total_loss, cer, wer))
    train_dataloader.dataset.shuffle()