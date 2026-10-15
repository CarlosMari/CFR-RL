# CFR-RL in PyTorch: toward topology generalization

This fork ports [CFR-RL](https://github.com/jrayzhang6/CFR-RL) (Zhang et al., IEEE JSAC 2020) from
TensorFlow to **PyTorch** as the base for research on making the policy generalize to unseen network
topologies.

- `torch_model.py`: the policy and actor-critic models in PyTorch.
- `torch_train.py`: distributed training with `torch.multiprocessing`: a central agent and one
  worker per CPU core that exchange weights and experience through queues. Metrics are logged to
  Weights & Biases.
- `torch_ckpts/`, `model_weights.pth`: trained checkpoints for Abilene.

The ongoing generalization work (actor-critic updates, reward changes, imitation learning, layer
normalization) is on the [`Frontera2`](../../tree/Frontera2) branch.

```bash
python3 torch_train.py   # see config.py for the topology and traffic files
```

The original README follows.

---

# CFR-RL: Traffic Engineering with Reinforcement Learning in SDN

This is a Tensorflow implementation of CFR-RL as described in our paper:

Junjie Zhang, Minghao Ye, Zehua Guo, Chen-Yu Yen, H. Jonathan Chao, "[CFR-RL: Traffic Engineering With Reinforcement Learning in SDN](https://arxiv.org/abs/2004.11986)," in IEEE Journal on Selected Areas in Communications, vol. 38, no. 10, pp. 2249-2259, Oct. 2020, doi: 10.1109/JSAC.2020.3000371.

# Prerequisites

- Install prerequisites (test with Ubuntu 20.04, Python 3.8.5, Tensorflow v2.2.0, PuLP 2.3, networkx 2.5, tqdm 4.51.0)
```
python3 setup.py
```

# Training

- To train a policy for a topology, put the topology file (e.g., Abilene) and the traffic matrix file (e.g., AbileneTM) in `data/`, then specify the file name in config.py, i.e., topology_file = 'Abilene' and traffic_file = 'TM', and then run 
```
python3 train.py
```
- Please refer to `data/Abilene` for more details about topology file format. 
- In a traffic matrix file, each line belongs to a N*N traffic matrix, where N is the node number of a topology.
- Please refer to `config.py` for more details about configurations. 

# Testing

- To test the trained policy on a set of test traffic matrices, put the test traffic matrix file (e.g., AbileneTM2) in `data/`, then specify the file name in config.py, i.e., test_traffic_file = 'TM2', and then run 
```
python3 test.py
```

# Reference

Please cite our paper if you find our paper/code is useful for your work.

@ARTICLE{jzhang,
  author={J. {Zhang} and M. {Ye} and Z. {Guo} and C. -Y. {Yen} and H. J. {Chao}},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={CFR-RL: Traffic Engineering With Reinforcement Learning in SDN}, 
  year={2020},
  volume={38},
  number={10},
  pages={2249-2259},
  doi={10.1109/JSAC.2020.3000371}}
