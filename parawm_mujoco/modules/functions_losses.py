import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class MseLoss(nn.Module):
    def __init__(self, is_proprio):
        super().__init__()
        self.is_proprio = is_proprio
        
    def forward(self, obs_hat, obs, reduce=True):
        if self.is_proprio:
            res = obs_hat - obs
        else:
            to_uint8 = lambda x: (x.detach() * 255).to(torch.uint8)
            res = torch.where(
                to_uint8(obs) == to_uint8(obs_hat), 
                torch.zeros_like(obs), obs_hat - obs)
        loss = 0.5 * res.pow(2).flatten(2, -1)
        loss = loss.sum(dim=-1, keepdim=True)
        return loss.mean() if reduce else loss


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes - 1)
        bins = torch.linspace(lower_bound, upper_bound, num_classes)
        self.register_buffer('bins', bins, persistent=False)

    def forward(self, output, target, reduce=True):
        target = symlog(target.squeeze(-1))
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index - 1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clip(weight, min=0, max=1)
        weight = weight.unsqueeze(-1)
        
        lower = F.one_hot(index - 1, self.num_classes)
        upper = F.one_hot(index, self.num_classes)
        target_prob = (1 - weight) * lower + weight * upper

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1, keepdim=True)
        return loss.mean() if reduce else loss

    def decode(self, input):
        output = F.softmax(input, dim=-1) @ self.bins
        output = symexp(output).reshape(*input.shape[:-1], 1)
        return output
