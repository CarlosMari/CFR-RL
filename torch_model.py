import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import inspect
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
import torch.utils.tensorboard as tensorboard

class Model(nn.Module):
    def __init__(self, config, input_dims, action_dim, max_moves, master=True):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.action_dim = action_dim
        self.max_moves = max_moves
        self.model_name = f"{config.version}-{config.project_name}_{config.method}_{config.model_type}_{config.topology_file}_{config.traffic_file}"

        """
        if config.method == 'actor_critic':
            self.create_actor_critic_model(config)
        elif config.method == 'pure_policy':
            self.create_policy_model(config)
        """
        """self.lr_schedule = optim.lr_scheduler.ExponentialLR(
            optimizer=optim.RMSprop(self.parameters(), lr=config.initial_learning_rate),
            gamma=config.learning_rate_decay_rate,
        )"""

        if master:
            if config.method == 'actor_critic':
                ckpt_path = f'./torch_ckpts/{self.model_name}/actor_critic_model.pth'
            elif config.method == 'pure_policy':
                ckpt_path = f'./torch_ckpts/{self.model_name}/pure_policy_model.pth'

            # Check if the directory exists, and create it if not
            ckpt_dir = os.path.dirname(ckpt_path)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            # Load the PyTorch checkpoint
            self.ckpt = torch.load(ckpt_path)

            # Set up the TensorBoard summary writer
            log_dir = f'./logs/{self.model_name}'

            # Check if the directory exists, and create it if not
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            self.writer = tensorboard.SummaryWriter(log_dir)


    """ This 2 functions will be deleted
    def create_actor_critic_model(self, config):
        pass  # Implement the actor-critic model architecture in PyTorch

    def create_policy_model(self, config):
        pass  # Implement the policy model architecture in PyTorch"""

    def value_loss_fn(self, rewards, values):
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        advantages = rewards - values
        value_loss = advantages.pow(2).mean()
        return value_loss, advantages

    def policy_loss_fn(self, logits, actions, advantages, entropy_weight=0.01, log_epsilon=1e-12):
        # Review operations make sure they are correct
        batch_size, max_moves, action_dim = actions.size()

        # Reshape
        actions = actions.view(batch_size, max_moves, action_dim)

        # Calculate policy
        policy = torch.softmax(logits, dim=1) # Shape [batch_size, action_dim]

        assert policy.shape[0] == actions.shape[0] and advantages.shape[0] == actions.shape[0]

        # Calculate the entropy
        entropy = -torch.sum(policy* torch.log(policy + log_epsilon), dim=-1)
        entropy = entropy.unsqueeze(-1)

        policy = policy.unsqueeze(-1)

        policy_loss = torch.log(torch.clamp(torch.sum(actions*policy, dim=-2),
                                            min = log_epsilon))
        policy_loss = torch.sum(policy_loss, dim=-1, keepdim=True)

        policy_loss -= entropy_weight * entropy
        policy_loss = torch.sum(policy_loss)

        return policy_loss, entropy


    def actor_critic_train(self, inputs, actions, rewards, entropy_weight=0.01):
        raise NotImplementedError

    def train(self, inputs, actions, rewards, entropy_weight=0.01):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError

    def restore_ckpt(self, checkpoint=''):
        raise NotImplementedError

    def save_ckpt(self, _print=False):
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
        super(ActorCriticModel, self).__init__(config,input_dim,action_dim, max_moves, master=master)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.Conv2D_out = config.Conv2D_out  # Why are this parameters
        self.Dense_out = config.Dense_out
        self.max_moves = max_moves
        self.actor = nn.Sequential(
            nn.Conv2d(self.input_dim[0], self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[1] * self.input_dim[2], self.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(self.Dense_out, self.action_dim)
        )

        self.critic = nn.Sequential(
            nn.Conv2d(self.input_dim[0], self.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[1] * self.input_dim[2], self.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(self.Dense_out, 1)

        )

        #self.summary()




        if config.optimizer == 'RMSprop':
            self.actor_optimizer = optim.RMSprop(self.parameters(), lr=config.initial_learning_rate)
            self.critic_optimizer = optim.RMSprop(self.parameters(), lr=config.initial_learning_rate)
        elif config.optimizer == 'Adam':
            self.actor_optimizer = optim.Adam(self.parameters(), lr=config.initial_learning_rate)
            self.critic_optimizer = optim.Adam(self.parameters(), lr=config.initial_learning_rate)

        self.lr_scheduler_actor = ExponentialLR(self.actor_optimizer, gamma=config.learning_rate_decay_rate)
        self.lr_scheduler_critic = ExponentialLR(self.critic_optimizer, gamma=config.learning_rate_decay_rate)


    def forward(self, inputs):
        logits = self.actor(inputs)
        values = self.critic(inputs)
        return logits, values

    def train(self, inputs, actions, rewards, entropy_weight=0.01):
        # We make sure the vectors are pytorch vectors
        inputs = torch.tensor(inputs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)


        # We tell the model that it is training
        self.actor.train()
        self.critic.train()

        # Zero -> Gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # We call the forward function and calculate the loss
        logits, values = self(inputs)

        # We calculate loss and advantages
        value_loss, advantages = self.value_loss_fn(rewards, values)
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss, entropy = self.policy_loss_fn(logits, actions, advantages, entropy_weight)
        policy_loss.backward()

        self.actor_optimizer.step()

        # Value_loss, entropy, actor gradients and critic gradients.
        return value_loss.item(), entropy.item(), [param.grad for param in self.actor.parameters()], [param.grad for param in self.critic.parameters()]

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


class PolicyModel(Model):
    def __init__(self, config, input_dim, action_dim):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.config = self.config

        self.model = nn.Sequential(
            nn.Conv2d(self.input_dim[0], config.Conv2D_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(self.Conv2D_out * self.input_dim[1] * self.input_dim[2], config.Dense_out),
            nn.LeakyReLU(),
            nn.Linear(config.Dense_out, self.action_dim)
        )


        if config.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr_schedule.get_lr()[0])
        elif config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr_schedule.get_lr()[0])

    def forward(self, input):
        return self.model(input)

    def train(self):
        raise NotImplementedError

    def predict(self, input):
        raise NotImplementedError