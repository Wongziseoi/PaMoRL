import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AgentLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.layer = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()))
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))
        
    def forward(self, inp):
        x = self.layer(inp)
        x = self.head(x)
        return x


class Encoder(nn.Module):
    def __init__(self, width, in_ch, stem_ch, min_res, act):
        super().__init__()
        feature_width = width // 2
        channels = stem_ch

        backbone = [ConvLayer(in_ch, channels, act())]
        while True:
            out_channels = channels * 2
            backbone += [ConvLayer(
                channels, out_channels, act())]
            channels = out_channels
            feature_width //= 2

            if feature_width == min_res:
                break

        self.backbone = nn.Sequential(*backbone)
        self.out_ch = channels
        self.embed = self.out_ch * (min_res ** 2)

    def forward(self, x):
        shape = x.shape[:2]
        x = x.flatten(0, 1)  # (B L) C H W
        x = self.backbone(x)
        x = x.flatten(1, -1)
        x = x.unflatten(0, shape)  # B L (C H W)
        return x


class Decoder(nn.Module):
    def __init__(self, stoch, out_ch, in_ch, stem_ch, min_res, act):
        super().__init__()
        backbone = [Rearrange(stoch, out_ch, min_res, act())]

        channels = out_ch
        feat_width = min_res
        while True:
            if channels == stem_ch:
                break
            
            out_channels = channels // 2
            backbone += [TransposeConvLayer(
                channels, out_channels, act())]
            channels = out_channels
            feat_width *= 2

        backbone += [nn.ConvTranspose2d(
            channels, in_ch, 4, 2, 1)]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        shape = sample.shape[:2]
        obs_hat = self.backbone(sample)
        obs_hat = obs_hat.unflatten(0, shape)
        return torch.sigmoid(obs_hat)


class ProprioEncoder(nn.Module):
    def __init__(self, inp_size, hidden, num_layer, act):
        super().__init__()
        backbone = [FeedForwardLayer(inp_size, hidden, act())]
        for _ in range(num_layer):
            backbone += [FeedForwardLayer(hidden, hidden, act())]
        self.backbone = nn.Sequential(*backbone)
        self.embed = hidden

    def forward(self, x):
        x = self.backbone(x)
        return x


class ProprioDecoder(nn.Module):
    def __init__(self, stoch, size, hidden, num_layer, act):
        super().__init__()
        backbone = [FeedForwardLayer(stoch, hidden, act())]
        for _ in range(num_layer):            
            backbone += [FeedForwardLayer(hidden, hidden, act())]
        backbone += [nn.Linear(hidden, size)]
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        x = self.backbone(sample)
        return x


class Head(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()))
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))

    def forward(self, feat):
        x = self.backbone(feat)
        x = self.head(x)
        return x
    

class InpLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = FeedForwardLayer(
            inp_size, hidden, act())
        self.head = nn.Sequential(
            nn.Linear(hidden, size, bias=False),
            nn.LayerNorm(size))

    def forward(self, inp):
        x = self.backbone(inp)
        x = self.head(x)
        return x
    

class ImsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.backbone = GatingLayer(inp_size, act())
        self.head = nn.Linear(inp_size, size)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        x = self.backbone(inp)
        x = self.head(x)
        x = self.norm(x)
        return x


class ObsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.head = nn.Linear(inp_size, size, bias=False)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        x = self.head(inp)
        x = self.norm(x)
        return x


class MixingLayer(nn.Module):
    def __init__(self, inp_size, hidden, bias=True):
        super().__init__()
        self.mix = nn.Parameter(init_weight(1, inp_size))
        self.register_parameter("mix_weight", self.mix)
        self.layer = nn.Linear(inp_size, hidden, bias=bias)

    def forward(self, inp, last):
        mix_w = torch.sigmoid(self.mix + 1)
        x = mix_w * inp + (1 - mix_w) * last
        x = self.layer(x)
        return x
    

class GatingLayer(nn.Module):
    def __init__(self, hidden, act, pdrop=0.1):
        super().__init__()
        self.layer = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(pdrop)
        self.act = act

    def forward(self, inp):
        p = self.layer(inp)
        x, g = p.chunk(2, dim=-1)
        x = F.mish(g) * x
        x = self.drop(x)
        x = x + inp
        x = self.norm(x)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(inp_size, size, bias=False),
            nn.LayerNorm(size))
        self.act = act

    def forward(self, inp):
        x = self.layer(inp)
        x = self.act(x)
        return x


class Rearrange(nn.Module):
    def __init__(self, in_dim, out_ch, min_res, act):
        super().__init__()
        out_dim = out_ch * (min_res ** 2)
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

        self.channels = out_ch
        self.width = min_res
        self.size = (
            self.channels, self.width, self.width)

    def forward(self, x):
        x = self.layer(x)
        x = x.flatten(0, 1)
        x = x.unflatten(1, self.size)
        x = self.norm(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        x = self.conv(inp)
        x = self.norm(x)
        x = self.act(x)
        return x


class TransposeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        x = self.conv(inp)
        x = self.norm(x)
        x = self.act(x)
        return x


class BatchNorm1d(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(size, eps=eps)

    def forward(self, inp):
        if inp.dim() > 2:
            shape = inp.shape[:2]
            x = inp.flatten(0, 1)
            x = self.norm(x)
            x = x.unflatten(0, shape)
        else:
            x = self.norm(inp)
        return x
    

def init_weight(in_dim, out_dim):
    denoms = (in_dim + out_dim) / 2
    scale = 1 / denoms
    std = math.sqrt(scale) / 0.87962566103423978
    data = torch.randn(in_dim, out_dim) * std
    data = torch.clip(data, min=-2*std, max=2*std)
    return data


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
    return f
