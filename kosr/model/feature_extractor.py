import torch
import torch.nn as nn

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

class VGGExtractor(nn.Module):
    def __init__(self, feature_size=128, hidden_size=512, proj_dropout_rate=0.1):
        super(VGGExtractor,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, feature_size, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.post_extract_proj = nn.Linear(feature_size, hidden_size)
        self.dropout = nn.Dropout(p=proj_dropout_rate)
        
    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B,C,T,F = x.size()
        x = self.post_extract_proj(x.transpose(1,2).contiguous().view(B,T,C*F))
        x = self.dropout(x)
        if mask is None:
            return x, mask
        return x, compute_mask(mask)
        
class W2VExtractor(nn.Module):
    def __init__(self, feature_size=512, hidden_size=512, conv_dropout_rate=0.0, proj_dropout_rate=0.1):
        super(W2VExtractor,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, feature_size, 10, stride=5, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 3, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 2, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU(),
            nn.Conv2d(feature_size, feature_size, 2, stride=2, bias=False),
            nn.Dropout(p=conv_dropout_rate),
            nn.GELU()
        )
        self.post_extract_proj = nn.Linear(feature_size, hidden_size)
        self.dropout = nn.Dropout(p=proj_dropout_rate)
        
    def forward(self, x, mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        B,C,T,F = x.size()
        x = self.post_extract_proj(x.transpose(1,2).contiguous().view(B,T,C*F))
        x = self.dropout(x)
        if mask is None:
            return x, mask
        return x, compute_mask(mask)