from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def convbn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))

def convbn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))

def convgn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))

def convgn_group(in_channels, out_channels, groups, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(4, out_channels))


def convgn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))

def convgn_3d_group(in_channels, out_channels, groups, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, groups=groups, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.GroupNorm(4, out_channels))


def convbn_3d_1kk(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride),
                                   padding=(0, pad, pad), bias=False),
                         nn.BatchNorm3d(out_channels))


def convbn_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False),
                         nn.BatchNorm3d(out_channels))

def conv_3d_new(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), bias=False),
                         nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), bias=False))

def convTrans_3d_new(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                   padding=(pad, 0, 0), output_padding = (output_pad, 0, 0), bias=False),
                         nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, kernel_size, 1), stride=(1, stride, 1),
                                   padding=(0, pad, 0), output_padding = (0, output_pad, 0), bias=False),
                         nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, 1, kernel_size), stride=(1, 1, stride),
                                   padding=(0, 0, pad), output_padding = (0, 0, output_pad), bias=False))

def convbn_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1),
                         nn.BatchNorm3d(out_channels))

def conv_3d_dw(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False, groups=in_channels),
                         nn.Conv3d(in_channels, out_channels, kernel_size=1))

def convTrans_3d_dw(in_channels, out_channels, kernel_size, pad, output_pad, stride):
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1),
                         nn.ConvTranspose3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, output_padding=output_pad, bias=False, groups=out_channels))


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
