import torch
from tqdm import tqdm
# import multiprocessing as mp
import torch.multiprocessing as mp
import sys
from environment.game import CFRRL_Game

from models.actor_critic import ActorCriticModel
from algorithms.Reinforce import Reinforce
from environment.game import CFRRL_Game
from environment.env import Environment
from config import get_config
import numpy as np
from absl import app
from absl import flags
import random

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_agents', 1, 'number of agents')
flags.DEFINE_string('name', 'BLANK', 'name of the run')
flags.DEFINE_string('baseline', 'avg', 'avg: use average reward as baseline, best: best reward as baseline')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations each agent would run')
flags.DEFINE_integer('method',0,'0-> Policy, 1 -> Actor Critic')
flags.DEFINE_integer('tms',2000,'Number of TMS per agent')
FLAGS(sys.argv)


CHECK_GRADIENTS = True
WANDB_LOG = False
LOG_STEPS = 10 
METHOD = FLAGS.method
TMS = FLAGS.tms
if WANDB_LOG:
    import wandb


def central_agent(config, games, model_weight_queues, experience_queues):
    game = games[0]
    if METHOD == 1:
        print("Actor Critic Model")
        algorithm = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=True)
    else:
        print("Policy Model")
        algorithm = Reinforce(config, game.state_dims, game.action_dim, game.max_moves, master=True, name=FLAGS.name)
    
    algorithm.name = FLAGS.name
    # We initialize wandb
    if WANDB_LOG:
        wandb.init(
            # set the wandb project where this run will be logged
            project='CFR-RL',
            name= FLAGS.name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": algorithm.config.initial_learning_rate,
                "dataset": algorithm.config.topology_file,
                "TM": algorithm.config.traffic_file,
                "architecture": algorithm.config.model_type,
                "steps": algorithm.config.max_step,
                "agents": FLAGS.num_agents,
                "type": algorithm.config.model_type,
                "framework": "pytorch",
            }
        )

    algorithm.save_hyperparams(config)
    # network.restore_ckpt()

    # Initial step from checkpoint should be implemented
    #for step in tqdm(range(network.step, config.max_step), ncols=70, initial=network.step):
    # for step in tqdm(range(network.step, config.max_step)):
    for step in range(algorithm.step, config.max_step):
        algorithm.step += 1
        model_weights = algorithm.get_weights()
        for i in range(FLAGS.num_agents):
            model_weight_queues[i].put(model_weights)
            model_weight_queues[i].join()

        # ACTOR-CRITIC ALGORITHM
        if METHOD == 1:
            # Assemble experiences from the agents
            s_batch = []
            a_batch = []
            r_batch = []
            mat_batch = []
            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, mat_batch_agent = experience_queues[i].get()

                assert len(s_batch_agent) == FLAGS.num_iter, \
                    (len(s_batch_agent), len(a_batch_agent), len(r_batch_agent))

                # Convert lists to PyTorch tensors and concatenate
                s_batch += [s.clone().detach() for s in s_batch_agent]
                a_batch += [torch.tensor(a, dtype=torch.int64) for a in a_batch_agent]
                r_batch += [torch.tensor(r, dtype=torch.float32) for r in r_batch_agent]
                mat_batch += [torch.tensor(mat) for mat in mat_batch_agent]
            assert len(s_batch) * game.max_moves == len(a_batch)

            # Convert 'a_batch' to one-hot encoded tensors
            action_dim = game.action_dim
            actions = torch.zeros(len(a_batch), action_dim, dtype=torch.float32)
            actions.scatter_(1, torch.tensor(a_batch).unsqueeze(1), 1)
            value_loss, entropy = algorithm._train(s_batch, actions, r_batch, config.entropy_weight, mat_batch)


        # REINFORCE
        elif METHOD == 0:
            s_batch = []
            a_batch = []
            r_batch = []
            ad_batch = []
            #mat = game.get_topology()
            mats = []
            #print(mat)
            for i in range(FLAGS.num_agents):
                s_batch_agent, a_batch_agent, r_batch_agent, ad_batch_agent, mat_batch_agent = experience_queues[i].get()
                s_batch += [s.clone().detach() for s in s_batch_agent]
                a_batch += [torch.tensor(a, dtype=torch.int64) for a in a_batch_agent]
                r_batch += [torch.tensor(r, dtype=torch.float64) for r in r_batch_agent]
                ad_batch += [torch.tensor(ad, dtype=torch.float32) for ad in ad_batch_agent]
                mats += [mat.clone().detach() for mat in mat_batch_agent]

            assert len(s_batch) * game.max_moves == len(a_batch)



            action_dim = game.action_dim
            actions = torch.zeros(len(a_batch), action_dim, dtype=torch.float64)
            actions.scatter_(1, torch.tensor(a_batch).unsqueeze(1), 1)
            entropy = algorithm._train(s_batch, actions, np.vstack(ad_batch).astype(np.float32), config.entropy_weight, mats)

        #print(algorithm.conv_net.lr_scheduler.get_last_lr()[0])
        # Log training information - Should be moved to model.
        if WANDB_LOG and step % LOG_STEPS == 0:
            num_tms = step * FLAGS.num_agents * FLAGS.num_iter
            if METHOD == 1:
                actor_learning_rate = algorithm.lr_scheduler_actor.get_last_lr()[0]
                avg_value_loss = np.mean(value_loss)
                avg_reward = np.mean(r_batch)
                max_reward = np.max(r_batch)
                min_reward = np.min(r_batch)
                avg_entropy = torch.mean(entropy)

                wandb.log({
                    'learning_rate': actor_learning_rate,
                    'loss': avg_value_loss,
                    'reward': avg_reward,
                    'entropy': avg_entropy,
                    'max_reward': max_reward,
                    'min_reward': min_reward,
                    'tm_count': num_tms,
                }, step=step+1)
            else:
                rs = np.array(r_batch)
                better = np.sum(rs > 1)/ len(rs)
                worse = np.sum(rs < 1) / len(rs)
                worse_mean = np.mean(rs[rs < 1])
                try:
                    better_mean = np.mean(rs[rs > 1])
                except:
                    better_mean = 1
                avg_reward = np.mean(r_batch)
                avg_entropy = torch.mean(entropy)
                learning_rate = algorithm.conv_net.lr_scheduler.get_last_lr()[0]
                avg_advantage = np.mean(ad_batch)
                max_reward = np.max(r_batch)
                min_reward = np.min(r_batch)
                wandb.log({
                    'learning_rate': learning_rate,
                    'advantage': avg_advantage,
                    'reward': avg_reward,
                    'better': better,
                    'worse': worse,
                    'worse_mean': worse_mean,
                    'better_mean': better_mean,
                    'max_reward': max_reward,
                    'min_reward': min_reward,
                    'entropy': avg_entropy,
                    'tm_count': num_tms
                }, step=step+1)

        # Saves a checkpoint every n steps
        if step % config.save_step == config.save_step - 1:
            algorithm.save_ckpt(_print=True)
        if step % config.learning_rate_decay_step == 0 and step != 0:
            algorithm.step_scheduler()

def agent(agent_id, config, game, tm_subset, model_weight_queues, experience_queue):
    global_step = 0
    random_state = np.random.RandomState(seed=agent_id)

    if METHOD == 1:
        network = ActorCriticModel(config, game.state_dims, game.action_dim, game.max_moves, master=False)
    else:
        network = Reinforce(config, game.state_dims, game.action_dim, game.max_moves, master=False)
    print(f"Creating agent {agent_id}, using {network.device}")


    mat = torch.from_numpy(game.get_topology()).unsqueeze(0) # (C, H, W), (1, 12, 12)
    model_weights = model_weight_queues.get()
    model_weight_queues.task_done()
    network.load_state_dict(model_weights)

    idx = 0
    s_batch = []
    a_batch = []
    r_batch = []
    mat_batch = []
    if METHOD == 0:
        ad_batch = []
    run_iteration_idx = 0
    num_tms = len(tm_subset)
    run_iterations = FLAGS.num_iter

    while True:
        tm_idx = tm_subset[idx]
        state = torch.from_numpy(game.get_state(tm_idx)).permute(2,0,1) # C,H,W (1, 12, 12)
        s_batch.append(state) 

        with torch.no_grad():
            if METHOD== 1:
                _, _, policy = network(state, mat)
            else:
                # We add the batch dimension
                logits, policy = network(state.unsqueeze(0), mat.unsqueeze(0))

        assert np.count_nonzero(policy.numpy()[0]) >= game.max_moves, (policy, state)

        #actions = random_state.choice(game.action_dim, game.max_moves, p=policy.view(-1), replace=False)

        if global_step < 7500:
            if random.randint(0,10) < 5:
                """possibles = np.array(list(range(game.num_pairs)))
                actions = np.random.choice(possibles, size=game.max_moves, replace=False)"""
                actions = torch.multinomial(policy.view(-1), game.max_moves,replacement=False).numpy()
            else:
                actions = game.get_critical_topK_flows(tm_idx)

        else:
            actions = torch.multinomial(policy.view(-1), game.max_moves,replacement=False).numpy()
            if global_step == 7500:
                print('Done exploration')

        for a in actions:
            a_batch.append(a)

        mat_batch.append(mat)
        # Rewards
        reward = game.reward(tm_idx, actions)
        r_batch.append(reward)

        # Advantages
        if METHOD == 0:
            # advantages
            ad_batch.append(game.advantage(tm_idx, reward))
            game.update_baseline(tm_idx, reward)

        run_iteration_idx += 1
        if run_iteration_idx >= run_iterations:
            if METHOD == 1:
                #ac
                experience_queue.put([s_batch, a_batch, r_batch, mat_batch])
            elif METHOD == 0:
                #reinforce
                experience_queue.put([s_batch, a_batch, r_batch, ad_batch, mat_batch])

            model_weights = model_weight_queues.get()
            model_weight_queues.task_done()
            network.load_state_dict(model_weights)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]
            del mat_batch[:]

            if METHOD == 0:
                del ad_batch[:]

            run_iteration_idx = 0

        idx += 1

        if idx == num_tms:
            random_state.shuffle(tm_subset)
            idx = 0
        global_step += 1


def main(_):
    #torch.cuda.set_device(-1)  # Set an invalid device number
    torch.autograd.set_detect_anomaly(True)
    # Set the logging level
    #torch.backends.cudnn.benchmark = True # False  # Disable CUDA optimizations for deterministic behavior
    # torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    #torch.set_default_tensor_type(torch.FloatTensor)  # Set the default tensor type to CPU

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False

    config = get_config(FLAGS) or FLAGS

    #env = Environment(config,topology='topology_0', is_training=True)
    #game = CFRRL_Game(config, env)
    model_weights_queues = []
    experience_queues = []
    games = []
    if FLAGS.num_agents == 0 or FLAGS.num_agents >= mp.cpu_count():
        # FLAGS.num_agents = mp.cpu_count() - 1
        pass

        # FLAGS.num_agents = 1
    print(f'Number of agents: {FLAGS.num_agents + 1}, Number iterations: {FLAGS.num_iter}')

    for i in range(FLAGS.num_agents):
        env = Environment(config, topology=f'topology_{i+1}', is_training=True,N=TMS, seed=i+1)
        game = CFRRL_Game(config, env, baseline=METHOD==0)
        games.append(game)
        model_weights_queues.append(mp.JoinableQueue(1))
        experience_queues.append(mp.Queue(1))

    # Why not just create a bigger batch_size and use GPU
    tm_subsets = np.array_split(game.tm_indexes, FLAGS.num_agents)
    coordinator = mp.Process(target=central_agent, args=(config, games, model_weights_queues, experience_queues))

    coordinator.start()

    agents = []
    for i in range(FLAGS.num_agents):
        agents.append(mp.Process(target=agent,
                                 args=(i, config, games[i], tm_subsets[i], model_weights_queues[i], experience_queues[i])))

    for i in range(FLAGS.num_agents):
        agents[i].start()

    coordinator.join()


if __name__ == '__main__':
    app.run(main)
