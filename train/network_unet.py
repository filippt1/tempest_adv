import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import basicblock as B
import numpy as np

'''
# ====================
# Residual U-Net
# ====================
citation:
@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
# ====================
'''

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.act = nn.ReLU(inplace=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(h)
        return h + self.skip(x)


class UNetRes(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(UNetRes, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):

        # Resolve upsampling size issues with padding
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        # Forward UNet

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        # Crop result to original size
        x = x[..., :h, :w]

        return x



class UNetResTime(nn.Module):
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: list[int] = [64, 128, 256, 512],
        nb: int = 4,
        downsample_mode: str = 'strideconv',
        upsample_mode: str = 'convtranspose',
        bias: bool = True,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.m_head = nn.Conv2d(in_nc, nc[0], kernel_size=3, padding=1, bias=bias)

        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1_blocks = nn.ModuleList([TimeResBlock(nc[0], nc[0], time_emb_dim, bias=bias) for _ in range(nb)])
        self.m_down1_down = downsample_block(nc[0], nc[1], bias=bias, mode='2')

        self.m_down2_blocks = nn.ModuleList([TimeResBlock(nc[1], nc[1], time_emb_dim, bias=bias) for _ in range(nb)])
        self.m_down2_down = downsample_block(nc[1], nc[2], bias=bias, mode='2')

        self.m_down3_blocks = nn.ModuleList([TimeResBlock(nc[2], nc[2], time_emb_dim, bias=bias) for _ in range(nb)])
        self.m_down3_down = downsample_block(nc[2], nc[3], bias=bias, mode='2')

        self.m_body_blocks = nn.ModuleList([TimeResBlock(nc[3], nc[3], time_emb_dim, bias=bias) for _ in range(nb)])

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3_up = upsample_block(nc[3], nc[2], bias=bias, mode='2')
        self.m_up3_blocks = nn.ModuleList([TimeResBlock(nc[2], nc[2], time_emb_dim, bias=bias) for _ in range(nb)])

        self.m_up2_up = upsample_block(nc[2], nc[1], bias=bias, mode='2')
        self.m_up2_blocks = nn.ModuleList([TimeResBlock(nc[1], nc[1], time_emb_dim, bias=bias) for _ in range(nb)])

        self.m_up1_up = upsample_block(nc[1], nc[0], bias=bias, mode='2')
        self.m_up1_blocks = nn.ModuleList([TimeResBlock(nc[0], nc[0], time_emb_dim, bias=bias) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)

        x_in = torch.cat([x, condition], dim=1)
        x_in = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x_in)

        t_emb = self.time_mlp(time.view(time.shape[0]))

        x1 = self.m_head(x_in)

        x2 = x1
        for block in self.m_down1_blocks:
            x2 = block(x2, t_emb)
        x2 = self.m_down1_down(x2)

        x3 = x2
        for block in self.m_down2_blocks:
            x3 = block(x3, t_emb)
        x3 = self.m_down2_down(x3)

        x4 = x3
        for block in self.m_down3_blocks:
            x4 = block(x4, t_emb)
        x4 = self.m_down3_down(x4)

        x_body = x4
        for block in self.m_body_blocks:
            x_body = block(x_body, t_emb)

        x_up3 = self.m_up3_up(x_body + x4)
        for block in self.m_up3_blocks:
            x_up3 = block(x_up3, t_emb)

        x_up2 = self.m_up2_up(x_up3 + x3)
        for block in self.m_up2_blocks:
            x_up2 = block(x_up2, t_emb)

        x_up1 = self.m_up1_up(x_up2 + x2)
        for block in self.m_up1_blocks:
            x_up1 = block(x_up1, t_emb)

        x_out = self.m_tail(x_up1 + x1)

        x_out = x_out[..., :h, :w]
        return x_out


class UNetResEMA(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=True):
        super(UNetResEMA, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=bias, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=bias, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=bias, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=bias, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.ema3 = EMA(channels=nc[3])
        self.ema2 = EMA(channels=nc[2])
        self.ema1 = EMA(channels=nc[1])


        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=bias, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=bias, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=bias, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=bias, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=bias, mode='C')

    def forward(self, x0):

        # Resolve upsampling size issues with padding
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        # Forward UNet

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        x = self.m_body(x4)

        x = self.m_up3(x+self.ema3(x4))
        x = self.m_up2(x+self.ema2(x3))
        x = self.m_up1(x+self.ema1(x2))
        x = self.m_tail(x+x1)

        # Crop result to original size
        x = x[..., :h, :w]

        return x