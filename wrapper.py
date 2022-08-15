#-*- coding: utf-8 -*-
#File: wrapper.py
import gym
import numpy as np

class Timeout(gym.Wrapper):
  def __init__(self, env, max_timestep=3600):
    gym.Wrapper.__init__(self, env)
    self.max_timestep = max_timestep
    self.timestep = 0
    self.raw_env = env

  def step(self, action):
    self.timestep += 1
    obs, reward, done, info = self.env.step(action)
    info['timeout'] = False
    if self.timestep >= self.max_timestep:
      info['timeout'] = True
    return obs, reward, done, info
  
  def reset(self):
    self.timestep = 0
    return self.env.reset()

class EvalMetrics(gym.Wrapper):
  def __init__(self, env, number_of_cars):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.number_of_cars = number_of_cars

  def get_average_travel_time(self):
    return self.raw_env.world.eng.get_average_travel_time()

  def get_throughput(self):
    rest_cars = self.env.raw_env.world.eng.get_vehicles()
    throughput = self.number_of_cars - len(rest_cars)
    return throughput

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if info['timeout'] == True:
      info['average_travel_time'] = self.get_average_travel_time()
      info['throughput'] = self.get_throughput()
    return obs, reward, done, info
  
  def reset(self):
    return self.env.reset()

class ActionRepeat(gym.Wrapper):
  def __init__(self, env, repeat_times=10):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.repeat_times = repeat_times

  def step(self, action):
    reward_list = []
    for _ in range(self.repeat_times):
      obs, reward, done, info = self.env.step(action)
      reward_list.append(reward)
    reward = np.mean(reward_list, axis=0)
    return obs, reward, done, info
  
  def reset(self):
    return self.env.reset()

class CongestionEvent(gym.Wrapper):
  def __init__(self, env, max_congested_seconds=1800, difficulty=1, seed=None):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.max_congested_seconds = max_congested_seconds
    self.congested_road = None
    self.congested_seconds = 0
    self.difficulty = difficulty
    self.seed = seed

  def reset_event(self):
    if self.congested_road is not None:
      for road in self.congested_road:
        self.raw_env.world.eng.set_road_max_speed(road, 11.111)
    self.congested_road = None
    self.congested_seconds = 0

  def event(self):
    if self.congested_road is not None:
      for road in self.congested_road:
        self.raw_env.world.eng.set_road_max_speed(road, 11.111)
    self.congested_seconds = 0
    congested_road_id = np.random.randint(len(self.all_roads), size=self.difficulty)
    self.congested_road = [self.all_roads[road_id] for road_id in congested_road_id]
    for road in self.congested_road:
      if np.random.random() < 0.5:
        self.raw_env.world.eng.set_road_max_speed(road, 2.0)

  def step(self, action):
    self.congested_seconds += 1
    if (self.congested_seconds >= self.max_congested_seconds) or (self.congested_road is None):
      self.event()
    obs, reward, done, info = self.env.step(action)
    return obs, reward, done, info

  def reset(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    self.reset_event()
    obs = self.env.reset()
    return obs

class FrameStack(gym.Env):
  def __init__(self, env, hist_len, skip_len):
    gym.Wrapper.__init__(self, env)
    self.raw_env = self.env.raw_env
    self.hist_len = hist_len
    self.skip_len = skip_len
    self.obs_stack = []
    self.act_stack = []
    self.n_agents = self.raw_env.n_agents

  def _cat_obs(self, obs):
    ret_obs = []
    idx = len(self.obs_stack)
    for i in range(self.hist_len - 1):
      idx -= self.skip_len
      ret_obs.append(self.obs_stack[idx])
      ret_obs.append(self.act_stack[idx])
    ret_obs.append(obs)
    ret_obs.append(np.zeros((self.n_agents, 8)))
    ret_obs = np.concatenate(ret_obs, -1)
    return ret_obs

  def step(self, action):
    onehot_actions = np.zeros((self.n_agents, 8))
    for i, act in enumerate(action):
      onehot_actions[act] = 1
    self.act_stack.append(onehot_actions)
    next_obs, reward, done, info = self.env.step(action)
    ret_obs = self._cat_obs(next_obs)
    self.obs_stack.append(next_obs)
    return ret_obs, reward, done, info

  def reset(self):
    obs = self.env.reset()
    self.obs_stack = [np.zeros_like(obs) for _ in range(self.hist_len * self.skip_len)]
    self.act_stack = [np.zeros((self.n_agents, 8)) for _ in range(self.hist_len * self.skip_len)]
    ret_obs = self._cat_obs(obs)
    self.obs_stack.append(obs)
    return ret_obs
