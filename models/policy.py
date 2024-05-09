
from .torch_model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import ExponentialLR
from utils.weight_utils import weight_init
import json


class PolicyModel(Model):

    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(PolicyModel, self).__init__(config, input_dim, action_dim, max_moves, master=master)

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.config = config
        self.name = name

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * 12 * 12, 500),
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
            nn.Linear(288*2,288, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(288, action_dim, dtype=torch.float64)
        )
        # 288
        #self.action_layer = nn.Linear(288, action_dim, dtype=torch.float64)

        self.initialize_optimizers()

        self.to('cpu')
        if master:
            self.apply(weight_init)
            ckpt_path = f'./torch_ckpts/{self.name}/checkpoint_policy.pth'

            # Check if the directory exists, and create it if not
            self.ckpt_dir = os.path.dirname(ckpt_path)
            print(self.ckpt_dir)
            if os.path.exists(ckpt_path):
                print("Checkpoint has been restored")
                self.restore_ckpt(ckpt_path)
            else:
                if not os.path.exists(self.ckpt_dir):
                    os.makedirs(self.ckpt_dir)
                if not os.path.exists(ckpt_path):
                    self.step = 0
                    print(f'Path: {ckpt_path}')
                    print("There is no previous checkpoint, initializing...")
            print(f"Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

            torch.save(self.state_dict(),f'./torch_ckpts/{self.config.version}.pt')


    def to(self, device):
        self.device = device
        self.conv_block = self.conv_block.to(self.device)
        self.conv_block2 = self.conv_block2.to(self.device)
        self.mlp = self.mlp.to(self.device)

    def forward(self, inputs, mat):
        inputs = inputs.to(torch.float32).to(self.device)
        mat = mat.to(torch.float32).to(self.device)

        if len(mat.shape) == 1:
            mat = F.normalize(mat, dim=0)
            mat = mat.reshape(12, 12).unsqueeze(0).unsqueeze(0)
        else:
            mat = mat.reshape(mat.shape[0], mat.shape[1] * mat.shape[2])  # (B, 12^2)
            mat = F.normalize(mat, dim=1)
            mat = mat.reshape(mat.shape[0], 12, 12)  # (B,12,12)
            mat = mat.unsqueeze(0).permute(1, 0, 2, 3)  # (B,C,12,12)

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

    def initialize_optimizers(self):
        if self.config.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.config.initial_learning_rate)
        elif self.config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.config.initial_learning_rate)
        else:
            raise NotImplementedError

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay_rate)

    def step_scheduler(self):
        self.lr_scheduler.step()

    def _train(self, inputs, actions, advantages, entropy_weight, mats):

        mats = torch.stack(mats, dim=0).to(self.device)
        inputs = torch.stack(inputs, dim=0).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)

        # We tell the model that it is training
        self.train()

        self.optimizer.zero_grad()

        logits, policy = self(inputs.permute(0, 3, 1, 2), mats)

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)


        policy_loss.backward()


        self.optimizer.step()

        return entropy

    def restore_ckpt(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # print(checkpoint['critic_optimizer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Restoring Checkpoint.... Step: {self.step}")

    def save_ckpt(self, _print=True):
        # Create a directory for saving checkpoints if it doesn't exist
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # Save the model and other relevant information to a checkpoint file
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
        }

        # Define the checkpoint file path (e.g., checkpoint.pth)
        checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint_policy.pth')

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        if _print:
            print("Saved checkpoint for step {}: {}".format(self.step, checkpoint_path))
