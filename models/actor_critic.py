import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import inspect
from algorithms.BaseAlgo import Model
from torch.optim.lr_scheduler import ExponentialLR

class ActorCriticModel(Model):

    def __init__(self, config, input_dim, action_dim, max_moves, master=True):
        super(ActorCriticModel, self).__init__(config, input_dim, action_dim, max_moves, master=master)
        self.config = config
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.Conv2D_out = config.Conv2D_out 
        self.Dense_out = config.Dense_out
        self.max_moves = max_moves

        self.a_conv_block = nn.Sequential(
            nn.Conv2d(1, self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * 12 * 12, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 288),
            nn.LeakyReLU()
        )

        self.a_conv_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 288),
            nn.LeakyReLU(),
        )
        
        self.c_conv_block = nn.Sequential(
                    nn.Conv2d(1, self.Conv2D_out, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Flatten(),
                    nn.Linear(self.Conv2D_out * 12 * 12, 500),
                    nn.LeakyReLU(),
                    nn.Linear(500, 288),
                    nn.LeakyReLU()
                )

        self.c_conv_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 288),
            nn.LeakyReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(288*2,288, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(288, action_dim, dtype=torch.float64)
        )

        self.critic = nn.Sequential(
            nn.Linear(288*2,288, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(288, 1, dtype=torch.float64)
        )


        # We create the optimizers
        self.initialize_optimizers()

        if master:

            ckpt_path = f'./torch_ckpts/{self.model_name}/actor_critic_model.pth'

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

    def initialize_optimizers(self):
        if self.config.optimizer == 'RMSprop':
            self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.config.initial_learning_rate)
            self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=self.config.initial_learning_rate)
        elif self.config.optimizer == 'Adam':
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.initial_learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.initial_learning_rate)

        self.lr_scheduler_actor = ExponentialLR(self.actor_optimizer, gamma=self.config.learning_rate_decay_rate)
        self.lr_scheduler_critic = ExponentialLR(self.critic_optimizer, gamma=self.config.learning_rate_decay_rate)

    def step_scheduler(self):
        self.lr_scheduler_critic.step()
        self.lr_scheduler_actor.step()

    def forward(self, inputs, mat):
        # print("Input type: ")
        # print(type(inputs))
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
        x = self.a_conv_block(inputs)
        x_2 = self.a_conv_block2(mat)
        final_x = torch.cat((x, x_2), dim=1).to(torch.double)  # [B,288]

        x_c = self.c_conv_block(inputs)
        x_2_c = self.c_conv_block2(mat)
        final_x_c = torch.cat((x_c, x_2_c), dim=1).to(torch.double)  # [B,288]

        logits = self.actor(final_x)
        value = self.critic(final_x_c)
        if self.config.logit_clipping > 0:
            logits = self.config.logit_clipping * torch.tanh(logits)
        # Returns logits, policy
        policy = F.softmax(logits, dim=1)

        return logits, value, policy




        """logits = self.actor(inputs)
        logits = logits.to(torch.float64)
        if self.config.logit_clipping > 0:
            logits = self.config.logit_clipping * torch.tanh(logits)

        policy = F.softmax(logits, dim=1)
        values = self.critic(inputs)
        return logits, values, policy"""

    def _train(self, inputs, actions, rewards, entropy_weight=0.01, mats=None):
        # We make sure the vectors are pytorch vectors
        mats = torch.stack(mats, dim=0).to(self.device)
        inputs = torch.stack(inputs, dim=0)
        print('=======================')
        print(f'{mats.shape=}: {inputs.shape=}')
        print('=======================')
        # print(inputs.shape)
        # rewards = torch.tensor(rewards, dtype=torch.float64)
        # actions = torch.tensor(actions, dtype=torch.float64)

        # We tell the model that it is training
        self.actor.train()
        self.critic.train()

        # Zero -> Gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # We call the forward function and calculate the loss
        # [B, C_in, H, W]
        logits, values, policy = self(inputs.permute(0, 3, 1, 2), mats)

        # We calculate loss and advantages
        value_loss, advantages = self.value_loss_fn(rewards, values)
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
        policy_loss.backward()

        self.actor_optimizer.step()

        # Value_loss, entropy, actor gradients and critic gradients.
        # We return the gradients for assert
        return value_loss.item(), entropy
    
    def predict(self, input):
        """

        :param input:
        :return policy, values:
        """
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            logits, values = self.input()
            policy = F.softmax(logits, dim=1)
        return policy, values.item()

    def restore_ckpt(self, checkpoint_path=None):

        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint.pth')

        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        # print(checkpoint['critic_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Restoring Checkpoint.... Step: {self.step}")

    def save_ckpt(self, _print=True):
        # Create a directory for saving checkpoints if it doesn't exist
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # Save the model and other relevant information to a checkpoint file
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'step': self.step,
        }

        # Define the checkpoint file path (e.g., checkpoint.pth)
        checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint_ac.pth')

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

        if _print:
            print("Saved checkpoint for step {}: {}".format(self.step, checkpoint_path))

