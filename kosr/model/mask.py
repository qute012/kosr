import torch

def make_pad_mask(lengths, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))

    maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    
    return mask.unsqueeze(1)

def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)

def get_attn_pad_mask(lengths, xs=None, length_dim=-1):
    mask = ~make_pad_mask(lengths, length_dim)
    #mask = mask.expand(-1, xs, -1)
    mask = mask.unsqueeze(1).eq(0)
    return mask

def subsequent_mask(size, device="cpu", dtype=torch.bool):
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)

def target_mask(ys_in_pad, ignore_id):
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return (ys_mask.unsqueeze(-2) & m).unsqueeze(1).eq(0)

def compute_mask(mask):
    """Create a subsampled version of x_mask.
    Args:
        x_mask: Input mask (B, 1, T)
    Returns:
        x_mask: Output mask (B, 1, sub(T))
    """
    t1 = mask.size(2) - (mask.size(2) % 3)
    mask = mask[:, :, :t1][:, :, ::3]

    t2 = mask.size(2) - (mask.size(2) % 2)
    mask = mask[:, :, :t2][:, :, ::2]

    return mask