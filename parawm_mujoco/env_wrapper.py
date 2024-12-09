import logging
import gymnasium
from gymnasium.spaces import Box
from gymnasium.core import Env
import numpy as np
from dm_control import suite
from dm_env import specs
from collections import deque



class LastFrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for i in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        obs = self.obs_buffer[-1]
        return obs, total_reward, done, truncated, info
    

class MaxLastFrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info


class DeepMindControl(Env):
    metadata = {}
    
    def __init__(self, domain, task, is_proprio=True, seed=0, height=64, width=64, camera=None):
        self._env = suite.load(domain, task, task_kwargs={"random": int(seed)})
        self._is_proprio = is_proprio

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.height = height
        self.width = width

        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

        if self._is_proprio:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values())
        else:
            image_size = (height, width, 3)
            self._observation_space = Box(0, 255, image_size, dtype=np.uint8)
        self._action_space = _spec_to_box([self._env.action_spec()])
        self._reward_range = [-np.inf, np.inf]

    def __getattr__(self, name):
        """
        Add this here so that we can easily access
        attributes of the underlying env
        """
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def reward_range(self):
        return self._reward_range

    def step(self, action, return_obs=True):
        if action.dtype.kind == "f":
            action = action.astype(np.float32)
        assert self._action_space.contains(action)
        timestep = self._env.step(action)
        if return_obs:
            if self._is_proprio:
                observation = _flatten_obs(timestep.observation)
            else:
                observation = self.render()
            reward = timestep.reward
            termination = False  # we never reach a goal
            truncation = timestep.last()
            info = {"discount": timestep.discount}
            return observation, reward, termination, truncation, info
        else:
            reward = timestep.reward
            termination = False  # we never reach a goal
            truncation = timestep.last()
            info = {"discount": timestep.discount}
            return reward, termination, truncation, info


    def reset(self):
        timestep = self._env.reset()
        if self._is_proprio:
            observation = _flatten_obs(timestep.observation)
        else:
            observation = self.render()
        info = {}
        return observation, info

    def render(self):
        size = (self.height, self.width)
        return self._env.physics.render(*size, camera_id=self._camera)


def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)
