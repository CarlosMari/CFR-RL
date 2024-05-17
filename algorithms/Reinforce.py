
from .BaseAlgo import Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import ExponentialLR
from utils.weight_utils import weight_init
import json
from models.ConvNet import ConvNet
from models.AttentionPolicy import AttentionPolicy
from models.SimpleMLP import SimpleMLP

class Reinforce(Model):

    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(Reinforce, self).__init__(config, input_dim, action_dim, max_moves, master=master)

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.config = config
        self.name = name
        self.conv_net = ConvNet(config,input_dim, action_dim, max_moves, master=True, name=name)


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

            torch.save(self.conv_net.state_dict(),f'./torch_ckpts/{self.config.version}.pt')


    def to(self, device):
        self.conv_net.to(device)

    def __call__(self, input, mat):
        return self.conv_net(input, mat)

    def _train(self, inputs, actions, advantages, entropy_weight, mats):
        mats = torch.stack(mats, dim=0).to(self.device)
        inputs = torch.stack(inputs, dim=0).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)

        # We tell the model that it is training
        self.conv_net.train()

        self.conv_net.optimizer.zero_grad()

        logits, policy = self(inputs, mats)

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)


        policy_loss.backward()


        self.conv_net.optimizer.step()

        return entropy

    def restore_ckpt(self, checkpoint_path=None):
        self.conv_net.restore_ckpt(checkpoint_path)

    def save_ckpt(self, _print=True):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # Save the model and other relevant information to a checkpoint file
        checkpoint = {
            'model_state_dict': self.conv_net.state_dict(),
            'optimizer_state_dict': self.conv_net.optimizer.state_dict(),
            'step': self.step,
        }

        # Define the checkpoint file path (e.g., checkpoint.pth)
        checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint_policy.pth')

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        if _print:
            print("Saved checkpoint for step {}: {}".format(self.step, checkpoint_path))

