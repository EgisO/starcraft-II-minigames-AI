"""
Authors:
Egehan Orta 150160124
Yiğitcan Çoban 150160039
İlgin Balkan 150170901
Rumeysa Nur Arslan 150160804
"""

import gym
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
from itertools import product
from absl import flags


class CollectMineralShards(gym.Env):
    metadata = {'render.modes': ['human']}
    
    default_settings = {
        'map_name': "CollectMineralShards",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
        'realtime': False,
        'step_mul': 96
    }

    def __init__(self,**kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.env = None
        if not flags.FLAGS.is_parsed():
            flags.FLAGS([""])
        self.marines = []
        self.shards = []
        self.level = 0
        
        self.action_space = spaces.Discrete(8)
        
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape = (22,3)
        )

    def get_distances(self, unit, shard):
        return np.linalg.norm(np.array(unit).reshape((1,2)) - np.array(shard).reshape((1,2)), axis=1)

    def step(self, action):

        raw_obs = self.take_action(action) # take safe action
        reward = raw_obs.reward # get reward from the env
        obs = self.get_derived_obs(raw_obs) # get derived observation
        done = raw_obs.last()
             
        return obs, reward, done, {}  # return obs, reward and whether episode ends

    def take_action(self, action):
        # map value to action
        if len(self.marines) == 2:
            sel = action%2
            action_type = action//2
            
            # action_type:
            # 0 -> Selected Marine's closest
            # 1 -> Selected marine's furthest
            # 2 -> Other Marine's closest
            # 3 -> Other Marine's furthest
            
            m0_xy = (self.marines[0].x, self.marines[0].y)
            m1_xy = (self.marines[1].x, self.marines[1].y)

            m0_dist = [self.get_distances(m0_xy, (i.x, i.y)) for i in self.shards]
            m1_dist = [self.get_distances(m1_xy, (i.x, i.y)) for i in self.shards]

            m0_ = [np.argmin(m0_dist), np.argmax(m0_dist)]
            m1_ = [np.argmin(m1_dist), np.argmax(m1_dist)]
            

            if action_type == 0:
                if sel == 0:
                    xy = m0_[0]
                else:
                    xy = m1_[0]
            elif action_type == 1:
                if sel == 0:
                    xy = m0_[1]
                else:
                    xy = m1_[1]
            elif action_type == 2:
                if sel == 0:
                    xy = m1_[0]
                else:
                    xy = m0_[0]
            elif action_type == 3:
                if sel == 0:
                    xy = m1_[1]
                else:
                    xy = m0_[1]

            xy = (self.shards[xy].x, self.shards[xy].y)

            action_mapped = self.move(sel, xy)
        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move(self, midx, dest):
        try:
            selected = self.marines[midx]
            return actions.RAW_FUNCTIONS.Move_Move_pt("now", selected.tag, dest)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def reset(self):
        if self.env is None:
            self.init_env()
        self.marines = []
        self.shards = []
        self.level+=1
        raw_obs = self.env.reset()[0]
        yy = self.get_derived_obs(raw_obs)
        return yy

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)
    
    def get_derived_obs(self, raw_obs):
        obs = np.zeros((22,3), dtype=np.uint8)
        # 1 indicates my own unit, 3 indicates neutrals
        marines = self.get_units_by_type(raw_obs, 1)
        shards = self.get_units_by_type(raw_obs, 3)

        self.marines = []
        self.shards = []

        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i] = np.array([m.x, m.y, 2])
        for i, s in enumerate(shards):
            self.shards.append(s)
            obs[i+2] = np.array([s.x, s.y, len(shards)])
        return obs

    def get_units_by_type(self, obs, player_relative=0):
        """
        NONE = 0
        SELF = 1
        ALLY = 2
        NEUTRAL = 3
        ENEMY = 4
        """
        return [unit for unit in obs.observation.raw_units if unit.alliance == player_relative]

    def render(self, mode='human'):
        pass
    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

