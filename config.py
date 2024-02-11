class NetworkConfig(object):
    """
  Contains the necessary parameters/hyperparameters of the NN.
  """
    scale = 100

    max_step = 1000 * scale

    initial_learning_rate = 0.0001
    learning_rate_decay_rate = 0.96
    learning_rate_decay_step = 5 * scale
    moving_average_decay = 0.9999
    entropy_weight = 0.1

    save_step = 2.5 * scale
    max_to_keep = 1000

    Conv2D_out = 128
    Dense_out = 128

    optimizer = 'RMSprop'
    # optimizer = 'Adam'

    logit_clipping = 10  # 10 or 0, = 0 means logit clipping is disabled


class Config(NetworkConfig):
    version = '5.0.1_pre'

    project_name = 'CFR-RL'

    #method = 'actor_critic'
    method = 'pure_policy'

    model_type = 'Embed'

    topology_file = 'BSO'
    topology_file = 'Abilene'

    traffic_file = 'TM2'
    test_traffic_file = 'TM'

    num_topologies = 4

    tm_history = 1

    max_moves = 10  # percentage

    # For pure policy
    baseline = 'avg'  # avg, best


def get_config(FLAGS):
    config = Config

    for k, v in FLAGS.__flags.items():
        if hasattr(config, k):
            setattr(config, k, v.value)

    return config
