import torch
import torch.nn as nn

class JointNet(nn.module):
    def __init__(self, out_dim, hidden_dim=512):
        super(JointNet, self).__init__()
        self.joint = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, enc_state, dec_state):
        joint_state = torch.cat((enct_state, dec_state), dim=-1)
        joint_state = self.act(self.joint(joint_state))
        return self.fc(joint_state)
        