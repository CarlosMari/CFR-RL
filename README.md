# CFR-RL in PyTorch: generalizing across network topologies

This fork ports [CFR-RL](https://github.com/jrayzhang6/CFR-RL) (Zhang et al., IEEE JSAC 2020) from
TensorFlow to **PyTorch** and extends it toward a policy that works on network topologies it was not
trained on. CFR-RL learns which *critical flows* to reroute: the policy picks K source–destination
pairs, an LP routes them optimally, and every other flow stays on ECMP shortest paths.

**What this fork adds**

- **Multi-topology training:** each worker trains on a different topology (`resources/topologies/`:
  15 random internet-like 12-node graphs). `topology_0` (Abilene) is held out for testing.
- **Topology-aware policy** (`models/ConvNet.py`): two convolutional towers, one over the normalized
  traffic matrix and one over the link-capacity matrix, joined by an MLP that scores every
  source–destination pair.
- **Reward against a strong heuristic** (`environment/game.py`): the policy's max-link utilization is
  compared with the critical-top-K heuristic on the same traffic matrix.
- **REINFORCE with a per-traffic-matrix average baseline** (`algorithms/Reinforce.py`), trained with
  `torch.multiprocessing`: a central learner plus one worker per topology. Metrics go to Weights &
  Biases.
- Synthetic gravity-model traffic (`utils/traffic_generator.py`) and SLURM scripts for TACC Frontera
  (`config.slurm`, `runserial.bash`).

**Run**

```bash
pip install -r requirements.txt          # LPs need the GLPK solver (e.g. apt install glpk-utils)
python torch_train.py --num_agents 4 --name my-run
python pytorchtest.py --ckpt ./torch_ckpts/my-run/checkpoint_policy.pth   # evaluate on Abilene
```

Status: research code, work in progress. The policy is fixed to 12-node topologies.

The original README follows; its TensorFlow commands no longer apply to this fork.

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
