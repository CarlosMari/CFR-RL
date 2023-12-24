import torch
from absl import app
from env import Environment
from game import CFRRL_Game
from actor_critic import ActorCriticModel
from policy import PolicyModel
from config import get_config
from absl import flags
import torch
import wandb
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', './torch_ckpts/4.1.1-CFR-RL_pure_policy_Embed_Abilene_TM2\checkpoint_policy.pth', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

def sim(config, network, game):

        for tm_idx in tqdm(game.tm_indexes):
                state = game.get_state(tm_idx)
                mat = game.get_topology()
                if config.method == 'actor_critic':
                        network.actor.eval()
                        network.critic.eval()
                        _, _, policy = network(torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2),torch.from_numpy(mat.flatten()))
                else:
                        network.eval()
                        _, policy = network(torch.from_numpy(state).unsqueeze(0).permute(0, 3, 1, 2),torch.from_numpy(mat.flatten()))
                # Change it so it works with pure policy as well

                actions = policy[0].argsort()[-game.max_moves:].numpy()
                #print(actions.shape)
                game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay)

def main(_):
        torch.cuda.set_device(-1)
        config = get_config(FLAGS) or FLAGS
        env = Environment(config, is_training=False)
        game = CFRRL_Game(config, env)

        if config.method == 'actor_critic':
                network = ActorCriticModel(config,game.state_dims,game.action_dim, game.max_moves, master=False)
        else:
                network = PolicyModel(config, game.state_dims, game.action_dim, game.max_moves)

        wandb.init(
                # set the wandb project where this run will be logged
                project='CFR-RL-TEST',

                # track hyperparameters and run metadata
                config={
                        "dataset": network.config.topology_file,
                        "architecture": network.config.model_type,
                        "steps": network.config.max_step,
                        "method": network.config.method,
                        "framework": "Pytorch",
                        "test_matrix": network.config.test_traffic_file,
                        "train_matrix": network.config.traffic_file,
                }
        )
        _ = network.restore_ckpt(FLAGS.ckpt)


        # Simulates the network with given configuration
        sim(config, network, game)

if __name__ == '__main__':
    app.run(main)