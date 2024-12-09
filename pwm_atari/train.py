import os
import wandb
import colorama
import gymnasium
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import env_wrapper

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
from agents import ActorCriticAgent
from modules.world_models import ParallelWorldModel


permute = lambda x: x.permute(0, 3, 1, 2)[:, None]


def build_single_env(env_name, image_size, frame_skip, seed):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLastFrameSkipWrapper(env, skip=frame_skip)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, image_size, num_envs, frame_skip, seed):
    # lambda pitfall refs to: 
    # https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size, frame_skip):
        return lambda: build_single_env(env_name, image_size, frame_skip, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size, frame_skip) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(samples, world_model, agent, logger, total_steps):
    agent.eval()
    world_model.update(agent, *samples, logger, total_steps)


def train_agent_step(samples, world_model, agent, imagine_horizon, logger, total_steps):
    world_model.eval()
    imagine_outputs = world_model.imagine_data(
        agent, *samples, imagine_horizon, logger, total_steps)
    agent.update(*imagine_outputs, logger, total_steps)


def joint_train_world_model_agent(env_name, 
                                  max_steps,
                                  frame_skip,
                                  num_envs,
                                  image_size,
                                  replay_buffer,
                                  world_model, 
                                  agent,
                                  train_model_every_steps,
                                  train_agent_every_steps,
                                  agent_update,
                                  batch_size,
                                  batch_length,
                                  imagine_batch_size,
                                  imagine_context,
                                  imagine_horizon,
                                  save_every_steps, 
                                  seed, 
                                  logger,
                                  ):
    # create ckpt dir
    os.makedirs(f"ckpt/{args.n}", exist_ok=True)

    # build vec env, not useful in the Atari100k setting
    # but when the max_steps is large, you can use parallel envs to speed up
    vec_env = build_vec_env(env_name, image_size, num_envs, frame_skip, seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # world_model = torch.compile(world_model)
    # agent = torch.compile(agent)

    # reset envs and variables
    world_model.eval()
    agent.eval()
    state = world_model.initial(num_envs)
    is_first = np.zeros((num_envs, 1))
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()

    logger.log(f"Rollout/{env_name}_reward", 0, 0)
    logger.log("Rollout/buffer_length", 1, 1)

    # sample and train
    for total_steps in tqdm(range(max_steps // num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            with torch.no_grad():
                world_model.eval()
                agent.eval()
                feat, state = world_model.get_inference_feat(state, obs, is_first)
                env_action, action = agent.sample_as_env_action(feat, greedy=False)
                state = world_model.update_inference_state(state, action)
        else:
            if "Freeway" in env_name:
                env_action = np.ones((num_envs,), dtype=np.int64)
            else:
                env_action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(env_action)
        real_done = np.logical_or(done, info["life_loss"])
        replay_buffer.append(current_obs, env_action, reward, real_done, is_first)

        is_first = np.logical_or(real_done, truncated)
        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    if replay_buffer.ready():
                        logger.log(f"Rollout/{env_name}_reward", sum_reward[i], total_steps)
                        logger.log("Rollout/buffer_length", len(replay_buffer), total_steps)
                    sum_reward[i] = 0

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        buffer_ready = replay_buffer.ready()
        start_training = total_steps * num_envs >= 0
        train_model_interval = total_steps % (train_model_every_steps // num_envs) == 0
        train_agent_interval = total_steps % (train_agent_every_steps // num_envs) == 0

        models = (world_model, agent)
        logs = (logger, total_steps)
        if buffer_ready:
            samples = replay_buffer.sample(batch_size, batch_length)

            if train_model_interval:
                train_world_model_step(samples, *models, *logs)
            
            if train_agent_interval and start_training:
                train_agent_step(samples, *models, imagine_horizon, *logs)

        # save model per episode
        if total_steps % (save_every_steps // num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{args.n}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{args.n}/agent_{total_steps}.pth")


def build_world_model(conf, num_action, act, device):
    return ParallelWorldModel(conf.JointTrainAgent.VideoLogStep,
                              conf.BasicSettings.ObsShape,
                              num_action,
                              conf.Models.Stoch,
                              conf.Models.Discrete, 
                              conf.Models.Hidden,
                              conf.Models.WorldModel.Stem,
                              conf.Models.WorldModel.MinRes,
                              conf.Models.NumBin,
                              conf.Models.MaxBin,
                              conf.Models.WorldModel.DynScale,
                              conf.Models.WorldModel.RepScale,
                              conf.Models.WorldModel.ValScale,
                              conf.Models.WorldModel.KLFree,
                              conf.Models.Gamma ** conf.BasicSettings.FrameSkip,
                              conf.Models.Lambda,
                              conf.Models.Tau,
                              conf.Models.WorldModel.LR,
                              conf.Models.WorldModel.Eps,
                              conf.BasicSettings.UseAmp,
                              act, device,
                              ).to(device)


def build_agent(conf, num_action, act, device):
    return ActorCriticAgent(num_action,
                            conf.Models.Stoch * conf.Models.Discrete + \
                            conf.Models.Hidden,
                            conf.Models.Hidden,
                            conf.Models.Agent.EntropyCoef,
                            conf.Models.NumSample,
                            conf.Models.NumBin,
                            conf.Models.MaxBin,
                            conf.Models.Agent.MinPer,
                            conf.Models.Agent.MaxPer,
                            conf.Models.Agent.EMADecay,
                            conf.Models.Gamma ** conf.BasicSettings.FrameSkip,
                            conf.Models.Lambda,
                            conf.Models.Tau,
                            conf.Models.Agent.LR, 
                            conf.Models.Agent.Eps,
                            conf.BasicSettings.UseAmp,
                            act, device,
                            ).to(device)


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.benchmark = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=args.seed)
    wandb.init(
        project="Atari100K",
        group=f"{args.env_name}",
        name=f"PWM-{args.env_name}-seed{args.seed}"
    )
    logger = Logger()

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        dummy_env = build_single_env(args.env_name, 
                                     conf.BasicSettings.ObsShape[0], 
                                     conf.BasicSettings.FrameSkip, 
                                     args.seed,
                                     )
        num_action = dummy_env.action_space.n

        # build world model and agent
        act = getattr(nn, conf.Models.Act)
        world_model = build_world_model(conf, num_action, act, args.device)
        agent = build_agent(conf, num_action, act, args.device)

        # build replay buffer
        replay_buffer = ReplayBuffer(conf.BasicSettings.ObsShape,
                                     conf.JointTrainAgent.NumEnvs, 
                                     conf.JointTrainAgent.BufferMaxLength, 
                                     conf.JointTrainAgent.BufferWarmUp, 
                                     args.device,
                                     )

        # train
        joint_train_world_model_agent(args.env_name,
                                      conf.JointTrainAgent.SampleMaxSteps,
                                      conf.BasicSettings.FrameSkip,
                                      conf.JointTrainAgent.NumEnvs,
                                      conf.BasicSettings.ObsShape[0],
                                      replay_buffer, world_model, agent,
                                      conf.JointTrainAgent.TrainModelEverySteps,
                                      conf.JointTrainAgent.TrainAgentEverySteps,
                                      conf.JointTrainAgent.AgentUpdate,
                                      conf.JointTrainAgent.BatchSize,
                                      conf.JointTrainAgent.BatchLength,
                                      conf.JointTrainAgent.ImagineBatchSize,
                                      conf.JointTrainAgent.ImagineContext,
                                      conf.JointTrainAgent.ImagineHorizon,
                                      conf.JointTrainAgent.SaveEverySteps,
                                      args.seed, logger
                                      )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")
