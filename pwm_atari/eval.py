import gymnasium
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import colorama

from utils import seed_np_torch, load_config
import env_wrapper

permute = lambda x: x.permute(0, 3, 1, 2)[:, None]


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


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


def eval_episodes(num_episode, env_name, num_envs, frame_skip, seed, image_size, world_model, agent):
    world_model.eval()
    agent.eval()
    vec_env = build_vec_env(env_name, image_size, num_envs, frame_skip, seed)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    world_model.eval()
    agent.eval()
    state = world_model.initial(num_envs)
    is_first = np.ones((num_envs, 1))
    sum_reward = np.zeros(num_envs)
    obs, info = vec_env.reset()

    final_rewards = []
    while True:
        # sample part >>>
        with torch.no_grad():
            feat, state = world_model.get_inference_feat(state, obs, is_first)
            env_action, action = agent.sample_as_env_action(feat, greedy=False)
            state = world_model.update_inference_state(state, action)
    
        obs, reward, done, truncated, info = vec_env.step(env_action)
        real_done = np.logical_or(done, info["life_loss"])
        is_first = np.logical_or(real_done, truncated)
        # cv2.imshow("current_obs", process_visualize(obs[0]))
        # cv2.waitKey(10)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    final_rewards.append(sum_reward[i])
                    sum_reward[i] = 0
                    if len(final_rewards) == num_episode:
                        print("Mean reward: " + colorama.Fore.YELLOW + f"{np.mean(final_rewards)}" + colorama.Style.RESET_ALL)
                        return final_rewards

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.enabled = False

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, required=True)
    parser.add_argument("-env_name", type=str, required=True)
    parser.add_argument("-run_name", type=str, required=True)
    parser.add_argument("-device", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)

    # set seed
    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    dummy_env = build_single_env(args.env_name, 
                                 conf.BasicSettings.ObsShape[0], 
                                 conf.BasicSettings.FrameSkip, 
                                 args.seed)
    num_action = dummy_env.action_space.n
    act = getattr(nn, conf.Models.Act)
    world_model = train.build_world_model(conf, num_action, act, args.device)
    agent = train.build_agent(conf, num_action, act, args.device)
    root_path = f"ckpt/{args.run_name}"

    # print model params
    model_param = sum(p.numel() for p in world_model.parameters())
    agent_param = sum(p.numel() for p in agent.parameters())
    enc_param = sum(p.numel() for p in world_model.encoder.parameters())
    dec_param = sum(p.numel() for p in world_model.decoder.parameters())
    dyn_param = sum(p.numel() for p in world_model.dynamic.parameters())
    rnn_param = sum(p.numel() for p in world_model.dynamic.rnn_layer.parameters())
    head_param = sum(p.numel() for p in world_model.done_head.parameters())
    head_param += sum(p.numel() for p in world_model.reward_head.parameters())
    act_param = sum(p.numel() for p in agent.actor.parameters())
    val_param = sum(p.numel() for p in agent.critic.parameters())

    print("--"*20)
    print(f"Name: Model; Params: {(model_param / 1e6):.3f} M")
    print(f"Name: Agent; Params: {(agent_param / 1e6):.3f} M")
    print(f"Name: Encoder; Params: {(enc_param / 1e6):.3f} M")
    print(f"Name: Decoder; Params: {(dec_param / 1e6):.3f} M")
    print(f"Name: Dynamic; Params: {(dyn_param / 1e6):.3f} M")
    print(f"Name: RNN; Params: {(rnn_param / 1e6):.3f} M")
    print(f"Name: Reward & Done Head; Params: {(head_param / 1e6):.3f} M")
    print(f"Name: Actor; Params: {(act_param / 1e6):.3f} M")
    print(f"Name: Critic; Params: {(val_param / 1e6):.3f} M")
    print("--"*20)

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    print(steps)
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        episode_returns = eval_episodes(
            num_episode=50,
            env_name=args.env_name,
            num_envs=20,
            frame_skip=conf.BasicSettings.FrameSkip, 
            seed=args.seed,
            image_size=conf.BasicSettings.ObsShape[0],
            world_model=world_model,
            agent=agent
        )
        results.append([step, *episode_returns])
    
    if not os.path.exists("eval_result"):
        os.mkdir("eval_result")

    with open(f"eval_result/{args.run_name}.csv", "w") as fout:
        for result in results:
            line = ""
            for res in result:
                line += f"{res},"
            fout.write(line[:-1] + "\n")
