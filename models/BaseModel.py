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

    def restore_ckpt(self, checkpoint_path):
        """Load weights and optimizer state saved by Reinforce.save_ckpt; returns the saved step."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint.get('step', 0)
        print(f"Restoring checkpoint {checkpoint_path} (step {step})")
        return step
