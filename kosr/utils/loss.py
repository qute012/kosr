import torch
from torch import nn

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.
    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(self, size, padding_idx, smoothing=0.1, normalize_length=False, criterion=nn.KLDivLoss(reduction="none"),):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.
        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom

class AttentionJointCTC(nn.Module):
    def __init__(self, size, padding_idx, ctc_weight=0.3, smoothing=0.1, normalize_length=False, criterion=nn.KLDivLoss(reduction="none"),):
        """Construct an AttentionJointCTCLoss object."""
        super(AttentionJointCTC, self).__init__()
        self.ctc = nn.CTCLoss(blank=padding_idx, reduction='sum', zero_infinity=True)
        self.att = LabelSmoothingLoss(size, padding_idx, smoothing, normalize_length, criterion)
        self.cw = ctc_weight

    def forward(self, att_x, att_target, ctc_x, ctc_target, x_length, target_length):
        att_loss = self.att(att_x, att_target)
        ctc_loss = self.ctc(ctc_x, ctc_target, x_length, target_length)/ctc_x.size(1)
        return (1-self.cw)*att_loss + self.cw*ctc_loss
    
def build_criterion(conf):
    device = conf['setting']['device']
    loss_type = conf['setting']['loss_type']
    if loss_type=='label_smoothing':
        criterion = LabelSmoothingLoss(conf['model']['out_dim'], padding_idx=conf['model']['pad_id']).to(device)
    elif loss_type=='att_joint_ctc':
        criterion = AttentionJointCTC(conf['model']['out_dim'], padding_idx=conf['model']['pad_id']).to(device)
    elif loss_type=='cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=conf['model']['pad_id']).to(device)
    elif loss_type=='rnnt_loss':
        import warp_rnnt._C as core
        criterion = rnnt_loss
        
    return criterion