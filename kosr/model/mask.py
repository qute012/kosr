import torch

def make_pad_mask(lengths, xs=None, length_dim=-1):
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask.unsqueeze(1)

def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim)

def get_attn_pad_mask(lengths, xs=None, length_dim=-1):
    return ~make_pad_mask(lengths, xs, length_dim).unsqueeze(-2).lt(1)

def subsequent_mask(size, device="cpu", dtype=torch.bool):
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret).lt(1)

def target_mask(ys_in_pad, ignore_id):
    ys_mask = ys_in_pad != ignore_id
    m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
    return ys_mask.unsqueeze(-2) & m

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