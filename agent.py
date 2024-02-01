import numpy as np
import torch

from gym_env import GameEnv
from config import Config
import torch.multiprocessing as mp
from policy import PolicyModel
from torch_model import Model
from torch.distributions import Categorical
import wandb
from tqdm import tqdm

LOG_STEPS = 10


class Agent():
    model: Model

    def __init__(self, id: int, config: Config, game: GameEnv, model_weight_queue: mp.JoinableQueue,
                 experience_queue: mp.JoinableQueue, device: str = 'cpu'):
        self.id = id
        self.config = config
        self.game = game
        self.model_weight_queue = model_weight_queue
        self.experience_queue = experience_queue
        self.max_moves = self.config.max_moves
        self.create_network()
        self.device = device

        self.init_weights()
        print(f"Creating agent {self.id} using {self.device}")

    def set_seed(self):
        self.game.set_seed(self.id)

    def create_network(self):
        self.model = PolicyModel(self.config, self.game.traffic_matrices.shape, self.game.action_dim, self.max_moves)

    def run(self):
        # print(f"Running agent {self.id}")
        run_iterations = 10
        mat, tm = self.game.reset()
        state_batch = []

        action_batch = None
        reward_batch = []
        advantage_batch = []
        mat_batch = []
        idx = 0
        while True:
            state_batch.append(tm)
            logits, policy = self.model(torch.from_numpy(tm).unsqueeze(0).unsqueeze(0), torch.from_numpy(mat).unsqueeze(0).unsqueeze(0))
            actions = torch.multinomial(policy, self.game.max_moves, replacement=False)  # Shape [11,13]
            if action_batch is None:
                action_batch = actions
            else:
                action_batch = torch.cat((action_batch, actions))
            mat_batch.append(mat)
            # typing
            new_mat, new_tm, reward = self.game.step(actions)
            reward_batch.append(reward)

            # Skipping advantages - in the future will code it
            advantage_batch.append(self.game.advantage(reward))
            self.game.update_baseline(reward)

            if idx >= run_iterations:
                self.experience_queue.put(
                    [torch.from_numpy(np.array(state_batch)), action_batch, torch.from_numpy(np.array(reward_batch)),
                     torch.from_numpy(np.array(advantage_batch)), torch.from_numpy(np.array(mat_batch))])

                self.init_weights()
                state_batch = []
                action_batch = None
                reward_batch = []
                advantage_batch = []
                mat_batch = []

                idx = 0
            idx += 1
            tm = new_tm
            mat = new_mat

    def init_weights(self):
        model_weights = self.model_weight_queue.get()
        self.model_weight_queue.task_done()
        self.model.load_state_dict(model_weights)


class CentralAgent(Agent):
    def __init__(self, id, config, game, model_weight_queue, experience_queue, device='cpu', log=False, num_agents=12):
        super().__init__(id, config, game, model_weight_queue, experience_queue, device)
        self.log = log
        self.num_agents = num_agents
        self.model.save_hyperparams(self.config)


    def run(self):
        self.init_wandb()
        for step in tqdm(range(self.model.step, self.config.max_step), ncols=70, initial=self.model.step):
            self.model.step += 1
            model_weights = self.model.get_weights()

            # We distribute the weights
            for i in range(self.num_agents):
                self.model_weight_queue[i].put(model_weights)
                self.model_weight_queue[i].join()

            #print("Finished distributing weights")

            # We prepare lists with the data
            """s_batch = torch.empy(0)  # states
            a_batch = torch.empty(0)  # actions
            r_batch = torch.empty(0)  # rewards
            ad_batch = torch.empty(0)  # Advantages
            mat_batch = torch.empty(0)  # Topologies (part of state)"""

            # We get the data
            for i in range(self.num_agents):
                #print(f"Getting experience from {i}")
                s_agent, a_agent, r_agent, ad_agent, mat_agent = self.experience_queue[i].get()
                if i == 0:
                    s_batch = s_agent
                    a_batch = a_agent
                    r_batch = r_agent
                    ad_batch = ad_agent
                    mat_batch = mat_agent
                else:
                    s_batch = torch.cat((s_batch, s_agent))
                    a_batch = torch.cat((a_batch, a_agent))
                    r_batch = torch.cat((r_batch, r_agent))
                    ad_batch = torch.cat((ad_batch, ad_agent))
                    mat_batch = torch.cat((mat_batch, mat_agent))

                # s_batch += [s.clone().detach() for s in s_agent]
                # a_batch += [torch.tensor(a, dtype=torch.int64) for a in a_agent]
                # r_batch += [torch.tensor(r, dtype=torch.float64) for r in r_agent]
                # ad_batch += [torch.tensor(ad, dtype=torch.float32) for ad in ad_agent]
                # mat_batch += [torch.tensor(mat) for mat in mat_agent]

            assert len(s_batch) * self.game.max_moves == a_batch.shape[0] * a_batch.shape[
                1], f'{s_batch.shape}; {a_batch.shape}'

            # One hot encoding of actions
            #                 22, 13, 132
            # actions shape = [batch_size, max_moves, action_dim]
            one_hot_actions = torch.nn.functional.one_hot(a_batch.to(torch.int64), 132)

            log_dict = {}
            log_dict['entropy'] = self.model.backward(s_batch, one_hot_actions, ad_batch,
                                                      self.config.entropy_weight, mat_batch)

            log_dict['reward'] = torch.mean(r_batch)
            log_dict['advantage'] = torch.mean(ad_batch)
            log_dict['learning_rate'] = self.model.lr_scheduler.get_last_lr()[0]

            self.wandb_log(log_dict, step)

    def create_network(self):
        self.model = PolicyModel(self.config, self.game.traffic_matrices.shape, self.game.action_dim,
                                 self.game.max_moves, master=True)

    def init_wandb(self):

        if self.log:
            print("Initializing wandb")
            wandb.init(
                # set the wandb project where this run will be logged
                project='CFR-RL',

                # track hyperparameters and run metadata
                config={
                    "learning_rate": self.model.config.initial_learning_rate,
                    "dataset": self.model.config.topology_file,
                    "TM": self.model.config.traffic_file,
                    "architecture": self.model.config.model_type,
                    "steps": self.model.config.max_step,
                    "agents": self.num_agents,
                    "type": self.model.config.model_type,
                    "framework": "pytorch",
                }
            )
        else:
            print("There is no logging to wandb.")

    def wandb_log(self, log_dict, step):
        if self.log and step % LOG_STEPS == 0:
            wandb.log(log_dict, step=step + 1)

    def init_weights(self):
        pass
