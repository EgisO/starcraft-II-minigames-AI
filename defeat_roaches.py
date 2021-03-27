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


class DefeatRoaches(gym.Env):
    metadata = {'render.modes': ['human']}
    
    default_settings = {
        'map_name': "DefeatRoaches",
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
        self.roaches = []
        self.level = 0

        self.action_space = spaces.Discrete(25)
        
        self.observation_space = spaces.Box(
            low=0,
            high=64,
            shape = (100,3)
        )

    def get_distances(self, unit, shard):
        return np.linalg.norm(np.array(unit).reshape((1,2)) - np.array(shard).reshape((1,2)), axis=1)

    def step(self, action):
        raw_obs = self.take_action(action) # take safe action
        reward = raw_obs.reward # get reward from the env
        obs = self.get_derived_obs(raw_obs)

        done = raw_obs.last()
        return obs, reward, done, {}  # return obs, reward and whether episode ends

    def take_action(self, action):
        # map value to action
        # action:
        # 0 -> Attack self.roaches[0]
        # 1 -> Attack self.roaches[1]
        # 2 -> Attack self.roaches[2]
        # 3 -> Attack self.roaches[3]
        # 4 -> All marines run away
        # action > 4 -> self.marines[attack] run away

        m_x = np.mean([marine.x for marine in self.marines])
        m_y = np.mean([marine.y for marine in self.marines])

        r_x = np.mean([roach.x for roach in self.roaches])
        r_y = np.mean([roach.y for roach in self.roaches])

        if action < 4:
            action_mapped = self.attack(action)

        
        elif action == 4:

            d_x = m_x - r_x
            d_y = m_y - r_y
            norm = np.linalg.norm((d_x, d_y))
            move_x = d_x * 4 / norm
            move_y = d_y * 4 / norm

            move_x += m_x
            move_y += m_y

            action_mapped = self.move((int(move_x), int(move_y)))

        elif action > 4:
            

            d_x = m_x - r_x
            d_y = m_y - r_y
            norm = np.linalg.norm((d_x, d_y))
            move_x = d_x * 4 / norm
            move_y = d_y * 4 / norm

            move_x += m_x
            move_y += m_y

            rrr = action-5
            action_mapped = self.move_(rrr, (int(move_x), int(move_y)))

        # execute action
        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def attack(self, dest):
        # all unit attack dst
        try:
            return actions.RAW_FUNCTIONS.Attack_unit("now", [selected.tag for selected in self.marines], self.roaches[dest].tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def move_(self, mm, dest):
        # marine mm move to dest
        try:
            return actions.RAW_FUNCTIONS.Move_Move_pt("now", self.marines[mm].tag, dest)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def move(self, dest):
        # all unit move to dest
        try:
            return actions.RAW_FUNCTIONS.Move_Move_pt("now", [selected.tag for selected in self.marines], dest)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def reset(self):
        if self.env is None:
            self.init_env()
        self.marines = []
        self.roaches = []
        self.level+=1
        raw_obs = self.env.reset()[0]
        yy = self.get_derived_obs(raw_obs)
        return yy

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)
    
    def get_derived_obs(self, raw_obs):
        obs = np.zeros((100,3), dtype=np.uint8)
        # 1 indicates my own unit, 4 indicates enemy's
        marines = self.get_units_by_type(raw_obs, 1)
        roaches = self.get_units_by_type(raw_obs, 4)

        self.marines = []
        self.roaches = []

        for i, s in enumerate(roaches):
            self.roaches.append(s)
            obs[i] = np.array([s.x, s.y, len(roaches)])
        for i, m in enumerate(marines):
            self.marines.append(m)
            obs[i+4] = np.array([m.x, m.y, len(marines)])
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
