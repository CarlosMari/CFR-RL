import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.weight_utils import weight_init
import torch.optim as optim
import os
from models.BaseModel import BaseModel

class ConvNet(BaseModel):
    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(ConvNet, self).__init__(config, input_dim, action_dim, max_moves, master, name)
        self.version = 'Conv'

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 288),
            nn.LeakyReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 288),
            nn.LeakyReLU(),
        )

        self.mlp = nn.Sequential(
            #nn.Linear(288*2,288, dtype=torch.float64),
            #nn.LeakyReLU(),
            nn.Linear(288*2, action_dim, dtype=torch.float64),
        )
        # 288
        #self.action_layer = nn.Linear(288, action_dim, dtype=torch.float64)

        self.initialize_optimizers()

        self.to('cpu')
        

    def to(self, device):
        self.device = device
        self.conv_block = self.conv_block.to(self.device)
        self.conv_block2 = self.conv_block2.to(self.device)
        self.mlp = self.mlp.to(self.device)

    def forward(self, inputs, mat):
        inputs = inputs.to(torch.float32).to(self.device)
        mat = mat.to(torch.float32).to(self.device)
        # INPUTS (B,1,12,12)
        # MAT (B,1,12,12)

        assert mat.shape == inputs.shape, f'{mat.shape},{inputs.shape}'

        x = self.conv_block(inputs)
        x_2 = self.conv_block2(mat)
        final_x = torch.cat((x, x_2), dim=1).to(torch.double)  # [B,288]

        logits = self.mlp(final_x)

        if self.config.logit_clipping > 0:
            logits = self.config.logit_clipping * torch.tanh(logits)
        # Returns logits, policy
        policy = F.softmax(logits, dim=1)

        return logits.cpu(), policy.cpu()



    '''def restore_ckpt(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # print(checkpoint['critic_optimizer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Restoring Checkpoint.... Step: {self.step}")
'''
