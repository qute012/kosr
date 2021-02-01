import torch
import torch.nn as nn

class JointNet(nn.Module):
    def __init__(self, out_dim, in_dim=1024, hidden_dim=512, dropout_rate=0.1):
        super(JointNet, self).__init__()
        self.joint = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, enc_state, dec_state):
        joint_state = torch.cat((enct_state, dec_state), dim=-1)
        joint_state = self.joint(joint_state)
        joint_state = self.act(joint_state)
        joint_state = self.dropout(joint_state)
        joint_state = self.norm(joint_state)
        return self.fc(joint_state)
        