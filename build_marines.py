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


class BuildMarines(gym.Env):
    metadata = {'render.modes': ['human']}
    
    default_settings = {
        'map_name': "BuildMarines",
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
        self.command_center = None
        self.idle_scv = []
        self.scvs = []
        self.barracks = []
        self.supply_depots = []
        self.minerals = []
        self.mineral_amount = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=100000,
            shape = (6,1)
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

        if action == 0: #send idle to mine
            action_mapped = self.idle_mine()

        elif action == 1: #build rax
            action_mapped = self.build_rax()

        elif action == 2: #build supply depot
            action_mapped = self.build_supply_depot()
        
        elif action == 3: #create marine 
            action_mapped = self.create_marine()

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def create_marine(self):
        # build marine from random rax
        try:
            rax_sel = self.barracks[np.random.randint(len(self.barracks))-1]
            return actions.RAW_FUNCTIONS.Train_Marine_quick("now",rax_sel.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def idle_mine(self):
        # all scvs to mine
        try:
            tags = [selected.tag for selected in self.idle_scv]
            if tags == []:
                return actions.RAW_FUNCTIONS.no_op()
            else:
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", tags, self.minerals[0].tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def build_rax(self):
        # build rax to the right side of command center
        try:
            x, y = self.command_center.x, self.command_center.y
            r = self.command_center.radius
            delta_x = np.random.randint(-5 * r, -4*r)
            delta_y = np.random.randint(-4 * r, 4 * r)
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now",self.scvs[0].tag,(x - delta_x,y + delta_y))
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self):
        # build rax to the right side of command center
        try:
            x, y = self.command_center.x, self.command_center.y
            r = self.command_center.radius
            delta_x = np.random.randint(-3 * r, -r)
            delta_y = np.random.randint(-4 * r, 4 * r)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now",self.scvs[0].tag,(x - delta_x,y + delta_y))
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def reset(self):
        if self.env is None:
            self.init_env()

        self.command_center = None 
        self.idle_scv = []
        self.scvs = []
        self.barracks = []
        self.supply_depots = []
        self.minerals = []
        self.mineral_amount = 0

        raw_obs = self.env.reset()[0]
        yy = self.get_derived_obs(raw_obs)
        return yy

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)
    
    def get_derived_obs(self, raw_obs):
        obs = np.zeros((6,1), dtype=np.uint32)
        self.command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter)[0]
        self.barracks       = self.get_units_by_type(raw_obs, units.Terran.Barracks)
        self.idle_scv       = self.get_idle_scv(raw_obs)
        self.minerals       = self.get_minerals(raw_obs)
        self.supply_depots  = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot)
        self.scvs           = self.get_units_by_type(raw_obs, units.Terran.SCV)
        self.mineral_amount = self.get_mineral_amount(raw_obs)

        obs[0] = len(self.minerals)
        obs[1] = len(self.scvs)
        obs[2] = len(self.barracks)
        obs[3] = len(self.idle_scv)
        obs[4] = len(self.supply_depots)
        obs[5] = self.mineral_amount

        return obs
    
    def get_mineral_amount(self, obs):
        #collected mineral count
        return obs.observation.player[features.Player.minerals]

    def get_minerals(self, obs):
        #available minerals to mine
        return [unit for unit in obs.observation.raw_units if unit.alliance == 3]

    def get_units_by_type(self, obs, u_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == u_type]

    def get_idle_scv(self, obs):
        try:
            return [unit for unit in obs.observation.raw_units if unit.unit_type == units.Terran.SCV and int(unit.order_length) == 0]
        except:
            return []

    def render(self, mode='human'):
        pass
    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    