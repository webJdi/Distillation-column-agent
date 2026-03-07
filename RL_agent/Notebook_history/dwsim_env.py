import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import sys

import clr
dwsim_path = r"C:\Users\sigma\AppData\Local\DWSIM" 
sys.path.append(dwsim_path)

clr.AddReference("DWSIM.Automation")
clr.AddReference("DWSIM.Interfaces")
clr.AddReference("DWSIM.GlobalSettings")

from DWSIM.Automation import Automation3
from DWSIM.GlobalSettings import Settings

class ADUEnvironment(gym.Env):
    def __init__(self, flowsheet_path):
        super(ADUEnvironment, self).__init__()
        
        # 1. Initialize DWSIM Automation
        self.interf = Automation3()
        self.flowsheet_path = flowsheet_path
        self.sim = self.interf.LoadFlowsheet(self.flowsheet_path)
        
        # 2. Define Action Space (What the RL agent can control)
        # Example: Controlling 1 parameter (e.g., Reflux Ratio between 0.5 and 5.0)
        self.action_space = spaces.Box(low=0.5, high=5.0, shape=(1,), dtype=np.float32)
        
        # 3. Define Observation Space (What the RL agent can see)
        # Example: Seeing 3 parameters (Top Temp, Bottom Temp, Naphtha Flow)
        # Normalized between 0 and 1 is best for neural networks
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reload the baseline flowsheet to clear any errors from the last episode
        self.sim = self.interf.LoadFlowsheet(self.flowsheet_path)
        
        # Get initial state
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # 1. Apply the action to the DWSIM model
        reflux_ratio = float(action[0])
        
        # You will need to find the exact string ID of your column or stream in DWSIM
        # Example: setting a specification value
        # self.sim.GetFlowsheetSimulationObject("Column1").SetPropertyValue("RefluxRatio", reflux_ratio)
        
        # 2. Run the DWSIM Solver
        errors = self.interf.CalculateFlowsheet4(self.sim)
        
        # 3. Check for convergence crashes
        if len(errors) > 0:
            # If the solver crashes, give a massive negative reward and end the episode
            return self._get_observation(), -100.0, True, False, {"error": "Solver crashed"}
            
        # 4. Get the new observation state
        obs = self._get_observation()
        
        # 5. Calculate Reward (The RL agent's goal)
        # Example: Reward = Amount of Naphtha produced - Penalty for using too much Reflux
        reward = self._calculate_reward(obs, reflux_ratio)
        
        # 6. Check termination conditions (e.g., episode length)
        terminated = False
        truncated = False 
        
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        # Read properties from DWSIM objects here
        # Example mock data:
        top_temp = 100.0 # self.sim.GetFlowsheetSimulationObject("Stream_Top").GetPropertyValue("Temperature")
        bot_temp = 350.0 
        naphtha_flow = 50.0
        
        # Normalize your observations here before returning!
        return np.array([top_temp/500, bot_temp/500, naphtha_flow/100], dtype=np.float32)

    def _calculate_reward(self, obs, action):
        # Define what "success" looks like mathematically
        naphtha_flow = obs[2] * 100
        reflux_penalty = action * 5 
        return naphtha_flow - reflux_penalty