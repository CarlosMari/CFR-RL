from models.BaseModel import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.q = nn.Linear(input_size, 4*input_size)
        self.k = nn.Linear(input_size, 4*input_size)
        self.v = nn.Linear(input_size, 4*input_size)
        self.out = nn.Linear(4*input_size, output_size)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        out = F.scaled_dot_product_attention(q,k,v)
        #out = torch.concat((out,x), dim=-1)
        return F.gelu(self.out(out))

class AttentionPolicy(BaseModel):
    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(AttentionPolicy, self).__init__(config, input_dim, action_dim, max_moves, master, name)
        self.version = 'Attention'
        self.attn_block_1 = AttentionBlock(288,288)
        self.attn_block_2 = AttentionBlock(288,288)
        self.attn_block_3 = AttentionBlock(288,288)
        self.attn_block_4 = AttentionBlock(288,288)
        self.linear = nn.Linear(288, action_dim)

        self.initialize_optimizers()

        self.to('cpu')
        

    def to(self, device):
        super(AttentionPolicy, self).to(device)
        self.device = device
        
    def forward(self, inputs, mat):
        # Mat -> (B, 1, 12, 12)
        # Inputs -> (B, 1, 12, 12) 

        conv_inputs = torch.concat((inputs.float(), mat.float()), dim=1)
        attention_input = conv_inputs.view(conv_inputs.shape[0], -1)
        #mat = mat.view(mat.shape[0], mat.shape[1],-1)
        #inputs = inputs.view(inputs.shape[0], inputs.shape[1],-1)
        #attention_input = torch.concat((inputs.float(), mat.float()), dim=2).squeeze(1)

        
        out = self.attn_block_1(attention_input)
        out = self.attn_block_2(out)
        out = self.attn_block_3(out)
        out = self.attn_block_4(out)
        logits = self.linear(out)
        policy = F.softmax(logits, dim=1)
        return logits.cpu(), policy.cpu()


