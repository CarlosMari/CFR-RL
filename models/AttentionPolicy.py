from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPolicy(BaseModel):
    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(AttentionPolicy, self).__init__(config, input_dim, action_dim, max_moves, master, name)
        self.version = 'Conv'

        self.q = nn.Linear(288, 288)
        self.k = nn.Linear(288, 288)
        self.v = nn.Linear(288, 288)

        self.linear = nn.Linear(288, action_dim)

        self.initialize_optimizers()

        self.to('cpu')
        

    def to(self, device):
        super(AttentionPolicy, self).to(device)
        self.device = device
        
    def forward(self, inputs, mat):
        # Mat -> (B, 1, 12, 12)
        # Inputs -> (B, 1, 12, 12) 
        mat = mat.view(mat.shape[0], mat.shape[1],-1)
        inputs = inputs.view(inputs.shape[0], inputs.shape[1],-1)
        print(inputs.shape)
        print(mat.shape)
        attention_input = torch.concat((inputs.float(), mat.float()), dim=2).squeeze(1)
        print(attention_input.shape)

        q = self.q(attention_input)
        k = self.k(attention_input)
        v = self.v(attention_input)
        out = F.scaled_dot_product_attention(q,k,v)
        logits = self.linear(out)
        policy = F.softmax(logits)
        return logits.cpu(), policy.cpu()

