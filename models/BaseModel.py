import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


class BaseModel(nn.Module):
    def __init__(self, config, input_dim, action_dim, max_moves, master=True, name='None'):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.config = config
        self.name = name

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
