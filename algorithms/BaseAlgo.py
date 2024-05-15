from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import inspect

from game import Game
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
# import torch.utils.tensorboard as tensorboard
import wandb
from torchsummary import summary

import torch.nn.init as init
class Model(nn.Module, ABC):
    def __init__(self, config, input_dims, action_dim, max_moves, master=True):
        super(Model, self).__init__()
        self.Conv2D_out = config.Conv2D_out  # Why are this parameters
        self.Dense_out = config.Dense_out
        self.config = config
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.max_moves = max_moves
        self.master = master
        self.model_name = f"{config.version}"
        self.device = 'cpu'


    def check_gradients(model):
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print("NaNs detected in gradients!")
                    return True
        return False

    @staticmethod
    def softmax_cross_entropy_with_logits(labels, logits, dim=-1):
        return (-labels * F.log_softmax(logits, dim=dim)).sum(dim=dim)

    @staticmethod
    def value_loss_fn(rewards, values):
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        advantages = rewards - values
        value_loss = advantages.pow(2).mean()
        return value_loss, advantages

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)  # You can choose another initialization method if you prefer
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)



    def to(self, device):
        self.device = device

    def policy_loss_fn(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-12):

        #print(f'Advantages shape {advantages.shape}') # AC -> (30,1), Pure Policy -> (30,1)
        # Review operations make sure they are correct
        actions = actions.view(-1, self.max_moves, self.action_dim)
        # Calculate policy
        policy = torch.softmax(logits, dim=1)  # Shape [batch_size, action_dim]

        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]

        # Calculate the entropy
        #entropy = F.cross_entropy(logits, policy)
        entropy = self.softmax_cross_entropy_with_logits(policy,logits)

        # entropy = nn.cross_entropy_loss(logits, policy)

        entropy = entropy.unsqueeze(-1)
        policy = policy.unsqueeze(-1)

        product = torch.matmul(actions.float(), policy).squeeze()
        # Ensures the minimum is log_epilon
        policy_loss = torch.log(
            torch.clamp(product, min=log_epsilon)
        )

        policy_loss = torch.sum(policy_loss, dim=1, keepdim=True).to(self.device)

        policy_loss = (torch.multiply(policy_loss, (-advantages).detach())).cpu()

        policy_loss = policy_loss - entropy_weight * entropy
        loss = torch.sum(policy_loss)

        return loss, entropy

    def get_weights(self):
        return self.state_dict()

    def _train(self, inputs, actions, rewards, entropy_weight=0.01):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError

    def restore_ckpt(self, checkpoint_path=None):
        raise NotImplementedError

    def save_ckpt(self, _print=True):
        raise NotImplementedError

    def inject_summaries(self, summary_dict, step):
        raise NotImplementedError

    def save_hyperparams(self, config):
        fp = self.ckpt_dir + '/hyper_parameters'

        hparams = {k: v for k, v in inspect.getmembers(config)
                   if not k.startswith('__') and not callable(k)}

        if os.path.exists(fp):
            f = open(fp, 'r')
            match = True
            for line in f:
                idx = line.find('=')
                if idx == -1:
                    continue
                k = line[:idx - 1]
                v = line[idx + 2:-1]
                if v != str(hparams[k]):
                    match = False
                    print('[!] Unmatched hyperparameter:', k, v, hparams[k])
                    break
            f.close()
            if match:
                return

            f = open(fp, 'a')
        else:
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            f = open(fp, 'w+')

        for k, v in hparams.items():
            f.writelines(k + ' = ' + str(v) + '\n')
        f.writelines('\n')
        print("Save hyper parameters: %s" % fp)
        f.close()


