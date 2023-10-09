import torch
from tqdm import tqdm
import multiprocessing as mp
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
flags.DEFINE_integer('num_agents', 20, 'number of agents')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseleine')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations each agent would run')
FLAGS(sys.argv)
CHECK_GRADIENTS = False

def central_agent(config, game, model_weight_queues, experience_queues):
    if config.method == 'actor_critic':
        network = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=True)
    else:
        network = PolicyModel(config, game.state_dims, game.action_dim, game.max_moves, master=True)

    network.save_hyperparams(config)
    #network.restore_ckpt()
    # Initial step from checkpoint should be implemented
    for step in tqdm(range(network.step, config.max_step), ncols=70, initial=0):
        network.step += 1
        FLAGS.num_agents = 11
        print(f"This is a test, step {step}, NUM_AGENTS {FLAGS.num_agents}")
        model_weights = network.get_weights()
        #print(f"Printing weights: {model_weights}")

        for i in range(FLAGS.num_agents):
            #print(f"Iteration {i}")
            model_weight_queues[i].put(model_weights)
            #print(model_weight_queues[i].get())
            #print(f"Weights for {i}")

        print("Finished uploading!")

        # I would like to implement this on the object not via ifs
        if config.method == "actor_critic":
            # Assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []

            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent = experience_queues[i].get()

                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                s_batch += s_batch_agent
                a_batch += a_batch_agent
                r_batch += r_batch_agent

            assert len(s_batch)*game.max_moves == len(a_batch)

            # Convert 'a_batch' to one-hot encoded tensors
            action_dim = game.action_dim
            actions = torch.zeros(len(a_batch), action_dim, dtype=torch.float32)
            actions.scatter_(1, torch.tensor(a_batch).unsqueeze(1), 1)

            value_loss, entropy, actor_gradients, critic_gradients = network.train(s_batch, actions,r_batch, config.entropy_weight)

            if CHECK_GRADIENTS: # Checks if gradients are NaN
                for g in actor_gradients:
                    assert not torch.isnan(g).any(), ('actor_gradients', s_batch, a_batch, r_batch, entropy)
                for g in critic_gradients:
                    assert not torch.isnan(g).any(), ('critic_gradients', s_batch, a_batch, r_batch, entropy)

            if step % config.save_step == config.save_step - 1:
                network.save_ckpt(_print=True)
            """    
                #log training information
                actor_learning_rate = network.lr_schedule(network.actor_optimizer.iterations.numpy()).numpy()
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                avg_entropy = np.mean(entropy)
            
                network.inject_summaries({
                    'learning rate': actor_learning_rate,
                    'value loss': avg_value_loss,
                    'avg reward': avg_reward,
                    'avg entropy': avg_entropy
                    }, step)
                print('lr:%f, value loss:%f, avg reward:%f, avg entropy:%f'%(actor_learning_rate, avg_value_loss, avg_reward, avg_entropy))
                """
        elif config.method == 'pure_policy':
            # Will be implemented in the future
            raise NotImplementedError


def agent(agent_id, config, game, tm_subset, model_weight_queues, experience_queue):
    random_state = np.random.RandomState(seed=agent_id)
    print(f"Creating agent {agent_id}")
    if config.method == 'actor_critic':
        network = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=False)
    else:
        network = PolicyModel(config, game.state_dims, game.action_dim, game.max_moves, master=False)

    # Initial synchronization of the model weights
    # I have to check the format of the weights in the queue

    print("Attempting to read!!!!!")
    model_weights = model_weight_queues.get()

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
        state = game.get_state(tm_idx)
        s_batch.append(state)

        with torch.no_grad():
            policy, _ = network.actor(torch.unsqueeze(state, 0))
        assert np.count_nonzero(policy.numpy()[0]) >= game.max_moves, (policy, state)
        actions = random_state.choice(game.action_dim, game.max_moves, p=policy, replace=False)
        for a in actions:
            a_batch.append(a)

        # Rewards
        reward = game.reward(tm_idx, actions)
        r_batch.append(reward)

        # Advantages
        # This ifs should be done via enums
        if config.method == 'pure_policy':
            raise NotImplementedError
        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            if config.method == 'actor_critic':
                experience_queue.put([s_batch,a_batch,r_batch])
            elif config.method == 'pure_policy':
                raise NotImplementedError

            model_weights = model_weight_queues.get()
            network.load_state_dict(model_weights)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            if config.method == 'pure_policy':
                raise NotImplementedError

            run_iteration_idx = 0

        idx += 1
        if idx == num_tms:
            random_state.shuffle(tm_subset)
            idx = 0


def main(_):
    torch.cuda.set_device(-1)  # Set an invalid device number

    # Set the logging level
    torch.backends.cudnn.benchmark = False  # Disable CUDA optimizations for deterministic behavior
    #torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.set_default_tensor_type(torch.FloatTensor)  # Set the default tensor type to CPU

    # Configure logging (you can customize this)
    import logging
    logging.basicConfig(level=logging.INFO)

    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=True)
    game = CFRRL_Game(config, env)
    model_weights_queues = []
    experience_queues = []
    if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
        FLAGS.num_agents = mp.cpu_count() - 1

        #FLAGS.num_agents = 1
    print('Agent num: %d, iter num: %d\n' % (FLAGS.num_agents + 1, FLAGS.num_iter))
    for _ in range(FLAGS.num_agents):
        #print(f"Creating queue for {_}")
        model_weights_queues.append(mp.Queue(1))
        experience_queues.append(mp.Queue(1))

    tm_subsets = np.array_split(game.tm_indexes, FLAGS.num_agents)
    coordinator = mp.Process(target=central_agent, args=(config,game,model_weights_queues, experience_queues))


    coordinator.start()

    agents = []
    for i in range(FLAGS.num_agents):
        agents.append(mp.Process(target=agent, args=(i,config,game,tm_subsets[i],model_weights_queues[i], experience_queues[i])))

    for i in range(FLAGS.num_agents):
        agents[i].start()

    coordinator.join()

if __name__ == '__main__':
    app.run(main)