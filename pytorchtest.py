import torch
from absl import app
from env import Environment
from game import CFRRL_Game
from torch_model import Model, ActorCriticModel, PolicyModel
from config import get_config
from absl import flags
import torch
FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', 'torch_ckpts/TE_v2-CFR-RL_actor_critic_Conv_Abilene_TM/checkpoint.pth', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

def sim(config, network, game):

        for tm_idx in game.tm_indexes:
                state = game.get_state(tm_idx)
                network.actor.eval()
                network.critic.eval()
                # Change it so it works with pure policy as well
                _,_, policy = network(torch.from_numpy(state).unsqueeze(0).permute(0,3,1,2))
                actions = policy[0].argsort()[-game.max_moves:].numpy()
                print(actions.shape)
                game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay)

def main(_):
        #torch.cuda.set_device(-1)
        config = get_config(FLAGS) or FLAGS
        env = Environment(config,is_training=True)
        game = CFRRL_Game(config, env)

        if config.method == 'actor_critic':
                network = ActorCriticModel(config,game.state_dims,game.action_dim, game.max_moves)
        else:
                raise NotImplementedError
        step = network.restore_ckpt(FLAGS.ckpt)


        # Simulates the network with given configuration
        sim(config, network, game)

if __name__ == '__main__':
    app.run(main)