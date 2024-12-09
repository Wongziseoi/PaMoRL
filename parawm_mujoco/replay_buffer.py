import pickle
import torch
import torch.nn.functional as F
import numpy as np


permute = lambda x: x.permute(0, 1, 4, 2, 3)


class ReplayBuffer:
    def __init__(self, 
                 obs_shape,
                 num_action,
                 num_envs, 
                 max_length=int(2e5), 
                 warmup_length=2500, 
                 device='cpu',
                 tempreture=100.0,
                 ):
        
        default = (max_length // num_envs, num_envs)
        self.obs_buffer = torch.empty(*default, *obs_shape, dtype=torch.uint8)
        self.action_buffer = torch.empty(*default, num_action, dtype=torch.float32)
        self.visit_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.reward_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.done_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.is_first_buffer = torch.empty(*default, 1, dtype=torch.float32)

        self.num_envs = num_envs
        self.max_length = max_length
        self.length = -1
        self.warmup_length = warmup_length
        self.tempreture = tempreture
        self.device = device

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, horizon):
        obs, action, reward, done, is_first = [], [], [], [], []
        assert batch_size > 0
        length = torch.arange(horizon)
        for i in range(self.num_envs):
            tot_len = self.length + 1 - horizon

            starts = torch.randint(tot_len, (batch_size // self.num_envs,))
            indexes = length[None, :] + starts[:, None]

            obs += [self.obs_buffer[indexes, i].to(self.device)]
            action += [self.action_buffer[indexes, i].to(self.device)]
            reward += [self.reward_buffer[indexes, i].to(self.device)]
            done += [self.done_buffer[indexes, i].to(self.device)]
            is_first += [self.is_first_buffer[indexes, i].to(self.device)]
        
        obs = permute(torch.cat(obs, dim=0).float() / 255) # [B T C H W]
        action = torch.cat(action, dim=0)
        reward = torch.cat(reward, dim=0)
        done = torch.cat(done, dim=0)
        is_first = torch.cat(is_first, dim=0)
        return obs, action, reward, done, is_first

    def append(self, obs, action, reward, done, is_first):
        self.length = (self.length + 1) % (self.max_length // self.num_envs)
        self.obs_buffer[self.length] = torch.tensor(obs)
        self.action_buffer[self.length] = torch.tensor(action)
        self.reward_buffer[self.length] = torch.tensor(reward).view(-1, 1)
        self.done_buffer[self.length] = torch.tensor(done).view(-1, 1)
        self.is_first_buffer[self.length] = torch.tensor(is_first).view(-1, 1)

    def __len__(self):
        return self.length * self.num_envs


class ProprioReplayBuffer:
    def __init__(self, 
                 obs_shape,
                 num_action,
                 num_envs, 
                 max_length=int(2e5), 
                 warmup_length=2500, 
                 device='cpu',
                 tempreture=100.0,
                 ):
        
        default = (max_length // num_envs, num_envs)
        self.obs_buffer = torch.empty(*default, obs_shape, dtype=torch.float32)
        self.action_buffer = torch.empty(*default, num_action, dtype=torch.float32)
        self.visit_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.reward_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.done_buffer = torch.empty(*default, 1, dtype=torch.float32)
        self.is_first_buffer = torch.empty(*default, 1, dtype=torch.float32)

        self.num_envs = num_envs
        self.max_length = max_length
        self.length = -1
        self.warmup_length = warmup_length
        self.tempreture = tempreture
        self.device = device

    def ready(self):
        return self.length * self.num_envs > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, horizon):
        obs, action, reward, done, is_first = [], [], [], [], []
        assert batch_size > 0
        length = torch.arange(horizon)
        for i in range(self.num_envs):
            tot_len = self.length + 1 - horizon

            starts = torch.randint(tot_len, (batch_size // self.num_envs,))
            indexes = length[None, :] + starts[:, None]

            obs += [self.obs_buffer[indexes, i].to(self.device)]
            action += [self.action_buffer[indexes, i].to(self.device)]
            reward += [self.reward_buffer[indexes, i].to(self.device)]
            done += [self.done_buffer[indexes, i].to(self.device)]
            is_first += [self.is_first_buffer[indexes, i].to(self.device)]
        
        obs = torch.cat(obs, dim=0)
        action = torch.cat(action, dim=0)
        reward = torch.cat(reward, dim=0)
        done = torch.cat(done, dim=0)
        is_first = torch.cat(is_first, dim=0)
        return obs, action, reward, done, is_first

    def append(self, obs, action, reward, done, is_first):
        self.length = (self.length + 1) % (self.max_length // self.num_envs)
        self.obs_buffer[self.length] = torch.tensor(obs)
        self.action_buffer[self.length] = torch.tensor(action)
        self.reward_buffer[self.length] = torch.tensor(reward).view(-1, 1)
        self.done_buffer[self.length] = torch.tensor(done).view(-1, 1)
        self.is_first_buffer[self.length] = torch.tensor(is_first).view(-1, 1)

    def __len__(self):
        return self.length * self.num_envs
