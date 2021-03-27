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

class CollectMineralsAndGas(gym.Env):
    metadata = {'render.modes': ['human']}
    
    default_settings = {
        'map_name': "CollectMineralsAndGas",
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
        self.refineries = []
        self.supply_depots = []
        self.minerals = []
        self.mineral_amount = 0
        self.gas_naked = []
        self.gas_amount = 0

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0,
            high=100000,
            shape = (8,1)
        )

    def get_distances(self, unit, shard):
        return np.linalg.norm(np.array(unit).reshape((1,2)) - np.array(shard).reshape((1,2)), axis=1)

    def step(self, action):
        raw_obs = self.take_action(action)# take safe action
        reward = raw_obs.reward# get reward from the env
        obs = self.get_derived_obs(raw_obs)
        done = raw_obs.last()

        return obs, reward, done, {} # return obs, reward and whether episode ends

    def take_action(self, action):

        if action == 0: #send idle to mine
            action_mapped = self.idle_mine()  

        elif action == 1: #collect gas
            action_mapped = self.idle_gas() 

        elif action == 2: #create scv
            action_mapped = self.create_scv() 
        
        elif action == 3: #build gas 
            action_mapped = self.build_gas() 

        elif action == 4: # build supply depot
            action_mapped = self.build_supply_depot() 
        
        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs


    def create_scv(self):
        try:
            return actions.RAW_FUNCTIONS.Train_SCV_quick("now",self.command_center.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def idle_mine(self):
        # all idle scvs to mining
        try:
            tags = [selected.tag for selected in self.idle_scv]
            if tags == []:
                return actions.RAW_FUNCTIONS.no_op()
            else:
                random_ref = self.refineries[np.random.randint(len(self.refineries))-1]
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", tags, random_ref.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def idle_gas(self):
        # all idle scvs to gas station
        try:
            tags = [selected.tag for selected in self.idle_scv]
            if tags == []:
                return actions.RAW_FUNCTIONS.no_op()
            else:
                random_mineral = self.minerals[np.random.randint(len(self.minerals))-1]
                return actions.RAW_FUNCTIONS.Harvest_Gather_SCV_unit("now", tags, random_mineral.tag)
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self):
        # build supply depot to available area
        try:
            random_scv = self.scvs[np.random.randint(len(self.scvs))-1]
            x, y = self.command_center.x, self.command_center.y
            r = self.command_center.radius
            delta_x = np.random.randint(-3 * r, -r)
            delta_y = np.random.randint(-2 * r, 2 * r)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now",random_scv.tag,(x - delta_x,y + delta_y))
        except:
            return actions.RAW_FUNCTIONS.no_op()

    def build_gas(self):
        # build gas station into gas resource
        try:
            if len(self.gas_naked) > 0:
                random_scv = self.scvs[np.random.randint(len(self.scvs))-1]
                random_naked_gas = self.gas_naked[np.random.randint(len(self.gas_naked))-1]
                return actions.RAW_FUNCTIONS.Build_Refinery_pt("now", random_scv.tag, random_naked_gas.tag)
            else:
                return actions.RAW_FUNCTIONS.no_op()
        except:
            return actions.RAW_FUNCTIONS.no_op()



    def reset(self):
        if self.env is None:
            self.init_env()

        self.command_center = None 
        self.idle_scv = []
        self.scvs = []
        self.refineries = []
        self.supply_depots = []
        self.minerals = []
        self.mineral_amount = 0
        self.gas_amount = 0
        self.gas_naked = []

        raw_obs = self.env.reset()[0]
        yy = self.get_derived_obs(raw_obs)
        return yy

    def init_env(self):
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)
    
    def get_derived_obs(self, raw_obs):

        obs = np.zeros((8,1), dtype=np.uint32)
        self.command_center = self.get_units_by_type(raw_obs, units.Terran.CommandCenter)[0]
        self.idle_scv       = self.get_idle_scv(raw_obs)
        self.minerals       = self.get_minerals(raw_obs)
        self.refineries     = self.get_gas_building(raw_obs)
        self.supply_depots  = self.get_units_by_type(raw_obs, units.Terran.SupplyDepot)
        self.scvs           = self.get_units_by_type(raw_obs, units.Terran.SCV)
        self.mineral_amount = self.get_mineral_amount(raw_obs)
        self.gas_amount     = self.get_gas_amount(raw_obs)
        self.gas_naked      = self.get_gas_naked(raw_obs)

        obs[0] = len(self.minerals)
        obs[1] = len(self.scvs)
        obs[2] = len(self.refineries)
        obs[3] = len(self.idle_scv)
        obs[4] = len(self.supply_depots)
        obs[5] = self.mineral_amount
        obs[6] = self.gas_amount
        obs[7] = len(self.gas_naked)

        return obs
    
    def get_mineral_amount(self, obs):
        return obs.observation.player[features.Player.minerals]

    def get_gas_amount(self,obs):
        return obs.observation.player[features.Player.vespene]

    def get_minerals(self, obs):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == units.Neutral.MineralField]

    def get_gas_naked(self,obs):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == units.Neutral.VespeneGeyser]

    def get_gas_building(self,obs):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == units.Terran.Refinery]

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

