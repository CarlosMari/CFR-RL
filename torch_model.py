import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import inspect
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
#import torch.utils.tensorboard as tensorboard
import wandb
from torchsummary import summary

class Model(nn.Module):
    def __init__(self, config, input_dims, action_dim, max_moves, master=True):
        super(Model, self).__init__()
        self.Conv2D_out = config.Conv2D_out  # Why are this parameters
        self.Dense_out = config.Dense_out
        self.config = config
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.max_moves = max_moves
        self.master = master
        self.model_name = f"{config.version}-{config.project_name}_{config.method}_{config.model_type}_{config.topology_file}_{config.traffic_file}"



    """ This 2 functions will be deleted
    def create_actor_critic_model(self, config):
        pass  # Implement the actor-critic model architecture in PyTorch

    def create_policy_model(self, config):
        pass  # Implement the policy model architecture in PyTorch"""

    def initialize_optimizers(self):
        raise NotImplementedError

    def value_loss_fn(self, rewards, values):
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        advantages = rewards - values
        value_loss = advantages.pow(2).mean()
        return value_loss, advantages

    def policy_loss_fn(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-12):
        # Review operations make sure they are correct
        actions = actions.view(-1, self.max_moves, self.action_dim)

        # Calculate policy
        policy = torch.softmax(logits, dim=1)  # Shape [batch_size, action_dim]

        #print(f'Policy : {policy.shape}, Actions: {actions.shape} Advantages: {advantages.shape}')

        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]

        # Calculate the entropy
        # REVIEW ENTROPY CALCULATION
        entropy = F.cross_entropy(logits, policy)
        #entropy = nn.cross_entropy_loss(logits, policy)
        entropy = entropy.unsqueeze(-1)
        policy = policy.unsqueeze(-1)

        #print(f' Actions shape: {actions.shape}')
        #print(f' Policy shape: {policy.shape}')

        product = torch.matmul(actions, policy).squeeze()
        # Ensures the minimum is log_epilon
        policy_loss = torch.log(
            torch.clamp(product, min=log_epsilon)
        )

        policy_loss = torch.sum(policy_loss, dim=1, keepdim=True)
        policy_loss = torch.multiply(policy_loss, (-advantages).detach())

        policy_loss -= entropy_weight * entropy
        policy_loss = torch.sum(policy_loss)

        return policy_loss, entropy

    def get_weights(self):
        return self.state_dict()


    def train(self, inputs, actions, rewards, entropy_weight=0.01):
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


class ActorCriticModel(Model):

    def __init__(self, config, input_dim, action_dim, max_moves, master=True):
        super(ActorCriticModel, self).__init__(config, input_dim, action_dim, max_moves, master=master)
        self.config = config
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.Conv2D_out = config.Conv2D_out  # Why are this parameters
        self.Dense_out = config.Dense_out
        self.max_moves = max_moves
        #print(f'Dimensions : {self.input_dim}')
        self.actor = nn.Sequential(
            nn.Conv2d(self.input_dim[2], self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[0] * self.input_dim[1], self.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(self.Dense_out, self.action_dim),
        )



        self.critic = nn.Sequential(
            nn.Conv2d(self.input_dim[2], self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[0] * self.input_dim[1], self.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(self.Dense_out, 1)

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
            self.actor_optimizer = optim.RMSprop(self.parameters(), lr=self.config.initial_learning_rate)
            self.critic_optimizer = optim.RMSprop(self.parameters(), lr=self.config.initial_learning_rate)
        elif self.config.optimizer == 'Adam':
            self.actor_optimizer = optim.Adam(self.parameters(), lr=self.config.initial_learning_rate)
            self.critic_optimizer = optim.Adam(self.parameters(), lr=self.config.initial_learning_rate)

        self.lr_scheduler_actor = ExponentialLR(self.actor_optimizer, gamma=self.config.learning_rate_decay_rate)
        self.lr_scheduler_critic = ExponentialLR(self.critic_optimizer, gamma=self.config.learning_rate_decay_rate)

    def forward(self, inputs):
        #print("Input type: ")
        #print(type(inputs))
        logits = self.actor(inputs)
        logits = logits.to(torch.float64)
        if self.config.logit_clipping > 0:
            logits = self.config.logit_clipping * torch.tanh(logits)
        logits = F.softmax(logits, dim=1)
        values = self.critic(inputs)
        return logits, values

    def train(self, inputs, actions, rewards, entropy_weight=0.01):
        # We make sure the vectors are pytorch vectors
        inputs = torch.stack(inputs, dim=0)
        #print(inputs.shape)
        #rewards = torch.tensor(rewards, dtype=torch.float64)
        #actions = torch.tensor(actions, dtype=torch.float64)

        # We tell the model that it is training
        self.actor.train()
        self.critic.train()

        # Zero -> Gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # We call the forward function and calculate the loss
        # [B, C_in, H, W]
        logits, values = self(inputs.permute(0,3,1,2))

        # We calculate loss and advantages
        value_loss, advantages = self.value_loss_fn(rewards, values)
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
        policy_loss.backward()

        self.actor_optimizer.step()

        # Value_loss, entropy, actor gradients and critic gradients.
        # We return the gradients for assert
        return value_loss.item(), entropy, [param.grad for param in self.actor.parameters()], [param.grad for
                                                                                                      param in
                                                                                                      self.critic.parameters()]

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
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.step = checkpoint['step']

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


class PolicyModel(Model):


    def __init__(self, config, input_dim, action_dim, max_moves, master=True):
        super(PolicyModel, self).__init__(config, input_dim, action_dim, max_moves, master=master)

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.config = self.config

        self.model = nn.Sequential(
            nn.Conv2d(self.input_dim[2], self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[0] * self.input_dim[1], self.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(self.Dense_out, self.action_dim),
        )

        if config.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.config.initial_learning_rate)
        elif config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.config.initial_learning_rate)

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.config.learning_rate_decay_rate)

        if master:

            ckpt_path = f'./torch_ckpts/{self.model_name}/Policy_model.pth'

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

    def forward(self, inputs):
        logits = self.model(inputs).to(torch.float64)
        if self.config.logit_clipping > 0:
            logits = self.config.logit_clipping * torch.tanh(logits)
        return F.softmax(logits, dim=1)


    def train(self, inputs, actions, advantages, entropy_weight):
        inputs = torch.stack(inputs, dim=0)
        advantages = torch.stack(advantages, dim=0)
        # print(inputs.shape)
        # rewards = torch.tensor(rewards, dtype=torch.float64)
        # actions = torch.tensor(actions, dtype=torch.float64)

        # We tell the model that it is training
        self.model.train()

        self.optimizer.zero_grad()

        logits = self(inputs.permute(0,3,1,2))

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
        policy_loss.backward()
        self.optimizer.step()

        return entropy





    def predict(self, input):
        raise NotImplementedError

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
