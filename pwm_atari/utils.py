import os
import wandb
import torch
import random
import numpy as np

from yacs.config import CfgNode as CN


def seed_np_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger():
    def __init__(self):
        self.tot_step = -1
        self.log_dict = {}

    def log(self, tag, value, step):
        if step > self.tot_step:
            wandb.log(self.log_dict, step=step)
            self.log_dict = {}
            self.tot_step = step
        self.log_dict.update({tag: value})
 
    def log_video(self, tag, value, step):
        value = torch.clip(value, min=0, max=1) * 255
        value = value.detach().to(torch.uint8).cpu().numpy()
        self.log(tag, [wandb.Video(value, fps=3, caption=f"step_{step}")], step)


class EMAScalar():
    def __init__(self, decay):
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar


def load_config(config_path):
    conf = CN()
    conf.Task = ""

    conf.BasicSettings = CN()
    conf.BasicSettings.Seed = 0
    conf.BasicSettings.ObsShape = None
    conf.BasicSettings.UseAmp = False
    conf.BasicSettings.FrameSkip = 0

    conf.Models = CN()
    conf.Models.Hidden = 0
    conf.Models.NumSample = 0
    conf.Models.NumBin = 0
    conf.Models.MaxBin = 0
    conf.Models.Act = ""
    conf.Models.Stoch = 0
    conf.Models.Discrete = 0
    conf.Models.Gamma = 1.0
    conf.Models.Lambda = 0.0
    conf.Models.Tau = 0.0

    conf.Models.WorldModel = CN()
    conf.Models.WorldModel.Stem = 0
    conf.Models.WorldModel.MinRes = 0
    conf.Models.WorldModel.DynScale = 0.0
    conf.Models.WorldModel.RepScale = 0.0
    conf.Models.WorldModel.ValScale = 0.0
    conf.Models.WorldModel.KLFree = 0.0
    conf.Models.WorldModel.LR = 0.0
    conf.Models.WorldModel.Eps = 0.0

    conf.Models.Agent = CN()
    conf.Models.Agent.EntropyCoef = 0.0
    conf.Models.Agent.MinPer = 0.0
    conf.Models.Agent.MaxPer = 0.0
    conf.Models.Agent.LR = 0.0
    conf.Models.Agent.Eps = 0.0
    conf.Models.Agent.EMADecay = 0.0

    conf.JointTrainAgent = CN()
    conf.JointTrainAgent.SampleMaxSteps = 0
    conf.JointTrainAgent.BufferMaxLength = 0
    conf.JointTrainAgent.BufferWarmUp = 0
    conf.JointTrainAgent.NumEnvs = 0
    conf.JointTrainAgent.BatchSize = 0
    conf.JointTrainAgent.BatchLength = 0
    conf.JointTrainAgent.ImagineBatchSize = 0
    conf.JointTrainAgent.ImagineContext = 0
    conf.JointTrainAgent.ImagineHorizon = 0
    conf.JointTrainAgent.TrainModelEverySteps = 0
    conf.JointTrainAgent.TrainAgentEverySteps = 0
    conf.JointTrainAgent.AgentUpdate = 0
    conf.JointTrainAgent.SaveEverySteps = 0
    conf.JointTrainAgent.VideoLogStep = 0

    conf.defrost()
    conf.merge_from_file(config_path)
    conf.freeze()

    return conf
