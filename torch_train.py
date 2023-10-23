import torch
from tqdm import tqdm
# import multiprocessing as mp
import torch.multiprocessing as mp
import sys
from game import CFRRL_Game
from torch_model import PolicyModel, ActorCriticModel
from game import CFRRL_Game
from env import Environment
from config import get_config
import numpy as np
from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_integer('num_agents', 1, 'number of agents')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseleine')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations each agent would run')
FLAGS(sys.argv)


CHECK_GRADIENTS = True
WANDB_LOG = True
LOG_STEPS = 10

if WANDB_LOG:
    import wandb


def central_agent(config, game, model_weight_queues, experience_queues):
    if config.method == 'actor_critic':
        print("Actor Critic Model")
        network = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=True)
    else:
        print("Policy Model")
        network = PolicyModel(config, game.state_dims, game.action_dim, game.max_moves, master=True)

    # We initialize wandb
    if WANDB_LOG:
        wandb.init(
            # set the wandb project where this run will be logged
            project='CFR-RL',

            # track hyperparameters and run metadata
            config={
                "learning_rate": network.config.initial_learning_rate,
                "dataset": network.config.topology_file,
                "architecture": network.config.model_type,
                "steps": network.config.max_step,
                "agents": FLAGS.num_agents,
                "type": network.config.model_type,
                "framework": "pytorch",
            }
        )

    network.save_hyperparams(config)
    # network.restore_ckpt()

    # Initial step from checkpoint should be implemented
    for step in tqdm(range(network.step, config.max_step), ncols=70, initial=network.step):
        network.step += 1
        model_weights = network.get_weights()
        for i in range(FLAGS.num_agents):
            model_weight_queues[i].put(model_weights)
            model_weight_queues[i].join()

        # ACTOR-CRITIC ALGORITHM
        if config.method == "actor_critic":
            # Assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()

                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                # Convert lists to PyTorch tensors and concatenate
                s_batch += [s.clone().detach() for s in s_batch_agent]
                a_batch += [torch.tensor(a, dtype=torch.int64) for a in a_batch_agent]
                r_batch += [torch.tensor(r, dtype=torch.float64) for r in r_batch_agent]

            assert len(s_batch) * game.max_moves == len(a_batch)

            # Convert 'a_batch' to one-hot encoded tensors
            action_dim = game.action_dim
            actions = torch.zeros(len(a_batch), action_dim, dtype=torch.float64)
            actions.scatter_(1, torch.tensor(a_batch).unsqueeze(1), 1)
            value_loss, entropy, actor_gradients, critic_gradients = network._train(s_batch, actions,
                                                                                   r_batch, config.entropy_weight)



            if CHECK_GRADIENTS:  # Checks if gradients are NaN
                for g in actor_gradients:
                    assert not torch.isnan(g).any(), ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in critic_gradients:
                    assert not torch.isnan(g).any(), ('critic_gradients', s_batch, a_batch, r_batch, entropy)

        # REINFORCE
        elif config.method == 'pure_policy':
            s_batch = []
            a_batch = []
            r_batch = []
            ad_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent = experience_queues[i].get()
                s_batch += [s.clone().detach() for s in s_batch_agent]
                a_batch += [torch.tensor(a, dtype=torch.int64) for a in a_batch_agent]
                r_batch += [torch.tensor(r, dtype=torch.float64) for r in r_batch_agent]
                ad_batch += [torch.tensor(ad, dtype=torch.float32) for ad in ad_batch_agent]

            assert len(s_batch) * game.max_moves == len(a_batch)

            action_dim = game.action_dim
            actions = torch.zeros(len(a_batch), action_dim, dtype=torch.float64)
            actions.scatter_(1, torch.tensor(a_batch).unsqueeze(1), 1)
            entropy, gradient = network._train(s_batch, actions, np.vstack(ad_batch).astype(np.float32), config.entropy_weight)

            if CHECK_GRADIENTS:  # Checks if gradients are NaN
                for g in gradient:
                    assert not torch.isnan(g).any(), ('GRADIENTS', s_batch, a_batch, r_batch, entropy)

        # Log training information - Should be moved to model.
        if WANDB_LOG and step % LOG_STEPS == 0:
            if config.method == "actor_critic":
                actor_learning_rate = network.lr_scheduler_actor.get_last_lr()[0]
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = torch.mean(entropy)

                wandb.log({
                    'learning_rate': actor_learning_rate,
                    'loss': avg_value_loss,
                    'reward': avg_reward,
                    'entropy': avg_entropy
                }, step=step+1)
            else:
                avg_reward = np.mean(r_batch)
                avg_entropy = torch.mean(entropy)
                learning_rate = network.lr_scheduler.get_last_lr()[0]
                avg_advantage = np.mean(ad_batch)
                wandb.log({
                    'learning_rate': learning_rate,
                    'advantage': avg_advantage,
                    'reward': avg_reward,
                    'entropy': avg_entropy
                }, step=step+1)

        # Saves a checkpoint every n steps
        if step % config.save_step == config.save_step - 1:
            network.save_ckpt(_print=True)
            #print(np.mean(value_loss))
        if step % config.learning_rate_decay_step == 0:
            network.step_scheduler()

def agent(agent_id, config, game, tm_subset, model_weight_queues, experience_queue):
    random_state = np.random.RandomState(seed=agent_id)
    print(f"Creating agent {agent_id}")
    if config.method == 'actor_critic':
        network = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=False)
    else:
        network = PolicyModel(config, game.state_dims, game.action_dim, game.max_moves, master=False)

    # Initial synchronization of the model weights
    # I have to check the format of the weights in the queue

    model_weights = model_weight_queues.get()
    model_weight_queues.task_done()
    network.load_state_dict(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []
    if config.method == 'pure_policy':
        ad_batch = []
    run_iteration_idx = 0
    num_tms = len(tm_subset)
    run_iterations = FLAGS.num_iter

    while True:
        tm_idx = tm_subset[idx]
        state = torch.from_numpy(game.get_state(tm_idx))
        s_batch.append(state)

        with torch.no_grad():
            # [B, C_in, H, W]
            extended_state = torch.unsqueeze(state, 0).permute(0, 3, 1, 2)

            if config.method == 'actor_critic':
                _, _, policy = network(extended_state)
            else:
                _, policy = network(extended_state)

        assert np.count_nonzero(policy.numpy()[0]) >= game.max_moves, (policy, state)

        actions = random_state.choice(game.action_dim, game.max_moves, p=policy.view(-1), replace=False)
        for a in actions:
            a_batch.append(a)

        # Rewards
        reward = game.reward(tm_idx, actions)
        r_batch.append(reward)

        # Advantages
        if config.method == 'pure_policy':
            # advantages
            ad_batch.append(game.advantage(tm_idx, reward))
            game.update_baseline(tm_idx, reward)

        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            if config.method == 'actor_critic':
                experience_queue.put([s_batch, a_batch, r_batch])
            elif config.method == 'pure_policy':
                experience_queue.put([s_batch, a_batch, r_batch, ad_batch])

            model_weights = model_weight_queues.get()
            model_weight_queues.task_done()
            network.load_state_dict(model_weights)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            if config.method == 'pure_policy':
                del ad_batch[:]

            run_iteration_idx = 0

        idx += 1

        if idx == num_tms:
            random_state.shuffle(tm_subset)
            idx = 0


def main(_):
    torch.cuda.set_device(-1)  # Set an invalid device number

    # Set the logging level
    torch.backends.cudnn.benchmark = False  # Disable CUDA optimizations for deterministic behavior
    # torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.set_default_tensor_type(torch.FloatTensor)  # Set the default tensor type to CPU

    config = get_config(FLAGS) or FLAGS

    env = Environment(config, is_training=True)
    game = CFRRL_Game(config, env)
    model_weights_queues = []
    experience_queues = []
    if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
        # FLAGS.num_agents = mp.cpu_count() - 1
        pass

        # FLAGS.num_agents = 1
    print(f'Number of agents: {FLAGS.num_agents + 1}, Number iterations: {FLAGS.num_iter}')

    for _ in range(FLAGS.num_agents):
        model_weights_queues.append(mp.JoinableQueue(1))
        experience_queues.append(mp.Queue(1))

    # Why not just create a bigger batch_size and use GPU
    tm_subsets = np.array_split(game.tm_indexes, FLAGS.num_agents)
    coordinator = mp.Process(target=central_agent, args=(config, game, model_weights_queues, experience_queues))

    coordinator.start()

    agents = []
    for i in range(FLAGS.num_agents):
        agents.append(mp.Process(target=agent,
                                 args=(i, config, game, tm_subsets[i], model_weights_queues[i], experience_queues[i])))

    for i in range(FLAGS.num_agents):
        agents[i].start()

    coordinator.join()


if __name__ == '__main__':
    app.run(main)
