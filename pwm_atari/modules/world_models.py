import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import OneHotCategorical

import modules.functions_losses as func
import modules.parallel_rnns as rnn
import modules.networks as net


params = lambda x: list(x.parameters())
permute = lambda x: x.permute(0, 3, 1, 2)
swap = lambda x: torch.transpose(x, 0, 1) 
ste_sample = lambda d: d.probs + (d.sample() - d.probs).detach()
to_param = lambda x: nn.Parameter(x)


class ParallelWorldModel(nn.Module):
    def __init__(self,
                 video_log,
                 obs_shape,
                 num_action,
                 stoch,
                 discrete,
                 hidden,
                 stem_ch,
                 min_res,
                 num_bin,
                 max_bin,
                 dyn_scale,
                 rep_scale,
                 val_scale,
                 kl_free,
                 gamma,
                 lambd,
                 tau,
                 lr,
                 eps,
                 use_amp,
                 act,
                 device,
                ):
        super().__init__()
        self.num_action = num_action
        self.hidden = hidden
        self.stoch_dim = stoch * discrete
        self.feat_dim = self.stoch_dim + hidden
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.val_scale = val_scale
        self.kl_free = kl_free
        self.gamma = gamma
        self.lambd = lambd
        self.tau = tau
        self.device = device
        self.batch_size = -1
        self.horizon = -1
        self.video_log = video_log

        self.device_type = "cuda" if "cuda" in device else "cpu"
        self.tensor_dtype = torch.float16 if use_amp else torch.float32
        self.use_amp = use_amp

        self.encoder = net.Encoder(obs_shape[0], obs_shape[-1], stem_ch, min_res, act)
        self.decoder = net.Decoder(
            self.stoch_dim, self.encoder.out_ch, obs_shape[-1], stem_ch, min_res, act)
        
        self.dynamic = PSSM(stoch, hidden, discrete, num_action, self.encoder.embed, act, device)
        self.done_head = net.Head(hidden, 1, hidden, act)
        self.reward_head = net.Head(hidden, num_bin, hidden, act)

        self.mse_loss = func.MseLoss()
        self.twohot_loss = func.SymLogTwoHotLoss(num_bin, -max_bin, max_bin)
        self.bce_logits_loss = F.binary_cross_entropy_with_logits

        model_params = params(self.dynamic) + params(self.done_head) + params(self.reward_head)
        vae_params = params(self.encoder) + params(self.decoder)

        self.optimizer = torch.optim.AdamW(model_params + vae_params, lr=lr, eps=eps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def preprocess(self, obs):
        tensor_obs = torch.tensor(
            obs, dtype=self.tensor_dtype, device=self.device) / 255
        tensor_obs = tensor_obs.permute(0, 3, 1, 2)[:, None] # [B, 1, C, H, W]
        return tensor_obs

    @torch.no_grad()
    def get_inference_feat(self, state, obs, is_first):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            embed = self.encoder(self.preprocess(obs)).squeeze(1)
            obs_stats = self.dynamic.suff_stats_layer("obs", embed)
            obs_stoch = ste_sample(self.dynamic.get_dist(obs_stats))

            is_first = torch.tensor(is_first, dtype=self.tensor_dtype, device=self.device)
            if is_first.sum() > 0:
                init_state = self.initial(obs_stoch.shape[0])
                for key, val in state.items():
                    num_axis = val.dim() - is_first.dim()
                    weight = is_first.unflatten(-1, [-1] + [1 for _ in range(num_axis)])
                    state[key] = val * (1 - weight) + init_state[key] * weight

            state.update({"stoch": obs_stoch, **obs_stats})
        return self.dynamic.get_feat(state), state
    
    @torch.no_grad()
    def update_inference_state(self, state, action):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            img_step_stats = self.dynamic.img_step(state, action, True)
            deter, _, para_stats, _ = img_step_stats
            state.update({"deter": deter, **para_stats})
        return state

    def initial(self, batch_size):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            return self.dynamic.initial(batch_size)
    
    def init_imagine_buffer(self, batch_size, horizon):
        if self.batch_size != batch_size or self.horizon != horizon:
            init_zeros = lambda s: torch.zeros(s, dtype=self.tensor_dtype, device=self.device)
            self.batch_size, self.horizon = batch_size, horizon

            deter_size = (batch_size, horizon+1, self.hidden)
            stoch_size = (batch_size, horizon+1, self.stoch_dim)
            action_size = (batch_size, horizon)
            self.deter_buffer = init_zeros(deter_size)
            self.stoch_buffer = init_zeros(stoch_size)
            self.action_buffer = init_zeros(action_size)
    
    @torch.no_grad()
    def get_video_frame(self, prior, index):
        stoch = self.dynamic.get_flatten_stoch(prior)
        pred_frame = self.decoder(stoch[index, None])
        return pred_frame

    @torch.no_grad()
    def imagine_data(self, agent, obs, action, reward, done, is_first, horizon, logger=None, step=None):
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            state, _, _, _ = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            img_state = {k: v.flatten(0, 1) for k, v in state.items()}
            batch_size = self.dynamic.get_feat(img_state).shape[0]
            self.init_imagine_buffer(batch_size, horizon)

            video_index, pred_video = torch.randint(batch_size, (1,), device=self.device), []

            for t in range(horizon):
                if logger is not None:
                    if step % self.video_log == 0:
                        pred_video += [self.get_video_frame(img_state, video_index)]
    
                self.deter_buffer[:, t] = self.dynamic.get_deter(img_state)
                self.stoch_buffer[:, t] = self.dynamic.get_flatten_stoch(img_state)
                self.action_buffer[:, t] = agent.sample(
                    torch.cat((self.deter_buffer[:, t], self.stoch_buffer[:, t]), dim=-1))
                img_state = self.dynamic.img_step(img_state, self.action_buffer[:, t])
            
            self.deter_buffer[:, -1] = self.dynamic.get_deter(img_state)
            self.stoch_buffer[:, -1] = self.dynamic.get_flatten_stoch(img_state)

            feat = torch.cat((self.deter_buffer, self.stoch_buffer), dim=-1)
            discount = (self.done_head(self.deter_buffer[:, 1:]) < 0) * self.gamma
            reward = self.twohot_loss.decode(self.reward_head(self.deter_buffer[:, 1:]))
            weight = torch.cat((torch.ones_like(reward[:, :1]), discount[:, :-1]), dim=1)
            
        if logger is not None:
            if step % self.video_log == 0:
                logger.log_video("Video/Imagination", torch.cat(pred_video, dim=1), step)

        return feat, self.action_buffer, discount, reward, weight

    def update(self, agent, obs, action, reward, done, is_first, logger=None, step=None):
        self.train()
        with torch.autocast(device_type=self.device_type, dtype=self.tensor_dtype, enabled=self.use_amp):
            post, prior, stoch, deter = self.dynamic.parallel_observe(self.encoder(obs), action, is_first)
            dyn_loss, rep_loss, real_kl, ent = self.dynamic.kl_loss(post, prior, self.kl_free)

            obs_hat = self.decoder(stoch)
            done_hat = self.done_head(deter)
            reward_hat = self.reward_head(deter)

            recon_loss = self.mse_loss(obs_hat, obs)
            done_loss = self.bce_logits_loss(done_hat, done)
            reward_loss = self.twohot_loss(reward_hat, reward)
            
            head_loss = done_loss + reward_loss
            model_loss = self.dyn_scale * dyn_loss + head_loss
            vae_loss = recon_loss + self.rep_scale * rep_loss

        self.scaler.scale(model_loss + vae_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if logger is not None:
            logger.log("WorldModel/recon_loss", recon_loss.item(), step)
            logger.log("WorldModel/reward_loss", reward_loss.item(), step)
            logger.log("WorldModel/dyn_loss", dyn_loss.item(), step)
            logger.log("WorldModel/rep_loss", rep_loss.item(), step)
            logger.log("WorldModel/real_kl", real_kl.item(), step)
            logger.log("WorldModel/vae_ent", ent.item(), step)

            if step % self.video_log == 0:
                video_index = torch.randint(obs.shape[0], (1,), device=self.device)
                logger.log_video("Video/Observation", obs[video_index], step)
                logger.log_video("Video/Reconstruction", obs_hat[video_index], step)


class PSSM(nn.Module):
    def __init__(self, stoch, hidden, discrete, action_dim, embed, act, device, unimix_ratio=0.01):
        super().__init__()
        self.stoch = stoch
        self.hidden = hidden
        self.discrete = discrete
        self.action_dim = action_dim
        self.unimix_ratio = unimix_ratio
        self.embed = embed
        self.act = act
        self.device = device
        self.num_rnns = 2

        stoch_dim = stoch * discrete
        inp_dim = stoch_dim + action_dim

        self.rnn_layer = self.init_cell()
        self.inp_layer = net.InpLayer(inp_dim, hidden, hidden, act)
        self.ims_stat_layer = net.ImsStatLayer(hidden, stoch_dim, act)
        self.obs_stat_layer = net.ObsStatLayer(embed, stoch_dim, act)
        self.one_hot = lambda x: F.one_hot(x.long(), action_dim).to(x.dtype)
        
        cell_ws = {}
        for id in range(self.num_rnns):
            cell_stats = self.rnn_layer[id].initial(1, id)
            cell_stats = {k: v.to(device) for k, v in cell_stats.items()}
            cell_ws.update(cell_stats)
        self.cell_ws = cell_ws
        self.init_deter = torch.zeros(1, hidden, requires_grad=False).to(device)
    
    def init_cell(self):
        layer_list = []
        for i in range(self.num_rnns):
            layer_list += [rnn.RNNCell(self.hidden, self.hidden, self.act)]
        layers = nn.ModuleList(layer_list)
        return layers

    @torch.no_grad()
    def initial(self, batch_size):
        init = {k: v.expand(batch_size, v.shape[-1]) 
                for k, v in self.cell_ws.items()}
        init_deter = self.init_deter.expand(
            batch_size, self.init_deter.shape[-1])
        init_logit, init_stoch = self.get_init_stoch(init_deter)
        init.update({
            "logit": init_logit,
            "stoch": init_stoch, 
            "deter": init_deter, 
        })
        return init
    
    def get_init_stoch(self, deter):
        stats = self.suff_stats_layer("ims", deter)
        dist = self.get_dist(stats)
        return stats["logit"], dist.mode
    
    def get_deter(self, state):
        return state["deter"]
    
    def get_feat(self, state):
        stoch = state["stoch"].flatten(-2, -1)
        return torch.cat((state["deter"], stoch), dim=-1)
    
    def get_flatten_stoch(self, state):
        return state["stoch"].flatten(-2, -1)
    
    def get_dist(self, state):
        probs = F.softmax(state["logit"], dim=-1)
        probs = probs * (1 - self.unimix_ratio) + \
            self.unimix_ratio / self.discrete
        return OneHotCategorical(probs=probs)
    
    def parallel_observe(self, embed, action, is_first):
        init = self.initial(action.shape[0])
        obs_stats = self.suff_stats_layer("obs", embed)
        oracle_stoch = ste_sample(self.get_dist(obs_stats))

        flatten_stoch = oracle_stoch.flatten(-2, -1)
        concat_input = torch.cat((flatten_stoch, self.one_hot(action)), dim=-1)
        latent, mask = self.inp_layer(concat_input), is_first
        deter, para_stats = self.cell_layers(latent, init, mask, True)

        ims_stats = self.suff_stats_layer("ims", deter[:, :-1])
        ims_stoch = ste_sample(self.get_dist(ims_stats))

        obs_stats = {k: v[:, 1:] for k, v in obs_stats.items()}
        obs_stoch = oracle_stoch[:, 1:]

        stats = {"deter": deter, **para_stats}
        stats = {k: v[:, :-1] for k, v in stats.items()}
        post = {"stoch": obs_stoch, **obs_stats, **stats}
        prior = {"stoch": ims_stoch, **ims_stats, **stats}
        return post, prior, flatten_stoch, deter

    def img_step(self, prev_state, prev_action, return_stats=False):
        prev_stoch = prev_state["stoch"].flatten(-2, -1)
        concat_input = torch.cat((prev_stoch, self.one_hot(prev_action)), dim=-1)
        deter, para_stats = self.cell_layers(
            self.inp_layer(concat_input), prev_state, None, False)
        
        ims_stats = self.suff_stats_layer("ims", deter)
        stoch = ste_sample(self.get_dist(ims_stats))
        
        if return_stats:
            return deter, stoch, para_stats, ims_stats
        else:
            prior = {
                "stoch": stoch, "deter": deter,
                **ims_stats, **para_stats}
            return prior

    def suff_stats_layer(self, name, x):
        if name == "ims":
            x = self.ims_stat_layer(x)
        elif name == "obs":
            x = self.obs_stat_layer(x)
        else:
            raise NotImplementedError
        
        logit = x.unflatten(-1, (self.stoch, self.discrete))
        return {"logit": logit}
    
    def cell_layers(self, input, state, is_first, is_parallel):        
        if is_parallel:
            deter, is_first = swap(input), swap(is_first)
        else:
            deter, is_first = input, is_first
        
        stats = {}
        for id, layer in enumerate(self.rnn_layer):
            deter, cell_stats = layer(
                deter, is_first, state, is_parallel, id)
            stats.update(cell_stats)

        if is_parallel:
            deter = swap(deter)
            stats = {k: swap(v) for k, v in stats.items()}
        return deter, stats

    def kl_loss(self, post, prior, free):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = kld(dist(post), dist(sg(prior)))
        dyn_loss = kld(dist(sg(post)), dist(prior))

        rep_loss = rep_loss.sum(dim=-1).mean()
        dyn_loss = dyn_loss.sum(dim=-1).mean()

        real_kl = dyn_loss
        ent = dist(post).entropy()
        ent = ent.sum(dim=-1).mean()

        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss, real_kl, ent
