"""
Authors:
Egehan Orta 150160124
Yiğitcan Çoban 150160039
İlgin Balkan 150170901
Rumeysa Nur Arslan 150160804
"""
import sys
from absl import flags
import ray
from ray.tune import run_experiments, register_env, Experiment

import tensorflow as tf
import os
from collect_mineral_shards import CollectMineralShards
from defeat_roaches import DefeatRoaches
from build_marines import BuildMarines
from collect_minerals_and_gas import CollectMineralsAndGas


FLAGS = flags.FLAGS
FLAGS(sys.argv)

env_name = 'my_env'
experiment_name = 'my_experiment'

ray.init()
register_env(env_name, lambda config: CollectMineralsAndGas())
experiment_spec = Experiment(
        experiment_name, #experiment name to log
        "DQN", #model to be used
        checkpoint_freq=100, #save model each 100th iteration
        stop={
            "training_iteration": 300, #stop model training after 300 iteration
        },
        config={ 
            "env": env_name,
            "framework": "tensorflow", # used framework
            "buffer_size": 50000,
            "timesteps_per_iteration": 1000, 
            "n_step": 3,
            "prioritized_replay": True,
            "grad_clip": None,
            "num_workers": 1,
            "num_gpus":1, # use gpu
            
            "exploration_config": {
                "type": "EpsilonGreedy", # use EpsilonGreedy for exploration
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 1000
            }
        },
    )

run_experiments(experiment_spec)

