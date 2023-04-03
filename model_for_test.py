import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
from math import ceil
from scipy import misc
# from model_utility import *

class Net_view(nn.Module):    
    def __init__(self, opt):        
        
        super(Net_view, self).__init__()
        num = opt.num_source
        an2 = opt.angular_out * opt.angular_out
        ### SAI ###
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels=num, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=num, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(5, 5), stride=(1, 1), padding=(6, 6),
                      dilation=(3, 3)),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=num, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(5, 5), stride=(1, 1), padding=(10, 10),
                      dilation=(5, 5)),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=num, out_channels=16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=an2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=an2, out_channels=an2, kernel_size=(5, 5), stride=(1, 1), padding=(14, 14),
                      dilation=(7, 7)),
        )

        self.ConvLinear = nn.Sequential(
            nn.Conv2d(in_channels=an2 * 4, out_channels=an2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=num, out_channels=an2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

        ### Lenslet
        self.up_conv = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(7, 3, 3), stride=(1, 1, 1), padding=(6, 1, 1),dilation=(2, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(6, 1, 1),dilation=(3, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(5, 2, 2), dilation=(5, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        # fuse
        self.Fuse = make_FuseBlock(layer_num=4, a=49, kernel_size=(3, 3))

        ### PSV ###
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 6, 6),
                      dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 4, 4),
                      dilation=(1, 2, 2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=32, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )

        self.residual_learning_2 = make_ResBlock_2(layer_num=4, an2=an2, kernel_size=(3, 3, 3))

        self.view_synthesis = nn.Sequential(
            nn.Conv3d(in_channels=an2, out_channels=an2, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )

        ### Concat
        self.lens_conv = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, ind_source, img_source, LFI, opt):
        
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out
        
        # ind_source 
        N,num_source,h,w = img_source.shape
        ind_source = torch.squeeze(ind_source)

        ### PSV ###
        D3_img = img_source.view(1, N, num_source, h, w)
        D3_img = self.feature_extraction(D3_img)
        D3_img = self.residual_learning_2(D3_img)
        D3_img = self.view_synthesis(D3_img)
        D3_img = D3_img.view(an2, N, h, w)

        ### SAI ###
        x0 = self.branch0(img_source)
        x1 = self.branch1(img_source)
        x2 = self.branch2(img_source)
        x3 = self.branch3(img_source)
        branch = torch.cat((x0, x1, x2, x3), dim=1)
        branch_target = self.ConvLinear(branch)
        short = self.shortcut(img_source)
        disp_target =branch_target + short

        #################### intermediate LF ##############################
        # Warping
        warp_img_input = img_source.view(N*num_source,1,h,w).repeat(an2,1,1,1)
        
        grid = []
        for k_t in range(0,an2):
            for k_s in range(0,num_source):
                ind_s = ind_source[k_s].type_as(img_source)
                ind_t = torch.arange(an2)[k_t].type_as(img_source)
                ind_s_h = torch.floor(ind_s/an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t/an)
                ind_t_w = ind_t % an   
                disp = disp_target[:,k_t,:,:]
                
                XX = torch.arange(0,w).view(1,1,w).expand(N,h,w).type_as(img_source)
                YY = torch.arange(0,h).view(1,h,1).expand(N,h,w).type_as(img_source)                 
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w-1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h-1) - 1.0                
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm),dim=3)
                grid.append(grid_t)
        grid = torch.cat(grid,0)
        warped_img = functional.grid_sample(warp_img_input,grid, align_corners=True).view(N,an2,num_source,h,w)

        ### Lenslet ###
        LFI = LFI.reshape(-1, 1, 1, 2, 2)
        LFI = LFI.permute(1, 2, 0, 3, 4)
        LFI = self.up_conv(LFI)
        LFI = torch.squeeze(LFI, 0)
        LFI = LFI.permute(1, 0, 2, 3)
        pixel_shuffle = nn.PixelShuffle(4)
        LFI = pixel_shuffle(LFI)
        lens_data = LFI.reshape(1, h, w, 8, 8)
        H = h
        W = w
        img = []
        sub = torch.zeros((1, H, W))
        for v in range(an):
            for u in range(an):
                sub[:, 0:H, 0:W] = lens_data[:, :, :, v, u]
                img.append(sub[:, :, :])
        img = torch.stack(img).cuda()
        img = img.permute(1, 0, 2, 3)
        img = self.Fuse(img)
        img = img.permute(1, 0, 2, 3)

        ### Concat ###
        warped_img_1 = warped_img.reshape(an2, 4, h, w)
        warped_0 = torch.cat((warped_img_1, D3_img, img), dim=1)
        warped_img = self.lens_conv(warped_0)
        warped_img = warped_img.view(1, an2, 6, h, w)
        return warped_img

class Net_refine(nn.Module):    
    def __init__(self, opt):        
        
        super(Net_refine, self).__init__()
        num_source = opt.num_source
        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out

        self.hang_conv = nn.Sequential(
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.lie_conv = nn.Sequential(
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=an, out_channels=an, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.residual_learning = make_ResBlock_1(layer_num=4, an2=an2, kernel_size=(3, 3, 3))
        self.residual_learning_1 = make_ResBlock(layer_num=2, an2=6, kernel_size=(3, 3, 3))
        self.residual_learning_3 = make_ResBlock_3(layer_num=2, an2=6, kernel_size=(3, 3, 3))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inter_lf, opt):
        
        N,an2,num_source,h,w = inter_lf.shape
        an = opt.angular_out
        ################# refine LF ###########################

        warped_img = self.residual_learning(inter_lf)
        warped_0 = warped_img.view(an2, 6, h, w)
        warped_h = warped_0.view(7, 7, 6, h, w)

        ### GM_H ###
        w_h = []
        for i in range(an):
            img = warped_h[i:i + 1, :, :, :, :]
            warped = self.hang_conv(img)
            w_h.append(warped)
        w_h = torch.stack(w_h, dim=0)
        w_h = w_h.view(1, an2, 6, h, w)
        w_h = torch.transpose(w_h, 1, 2)
        w_h = self.residual_learning_1(w_h)
        w_h = w_h.view(6, an2, h, w)
        w_h = torch.transpose(w_h, 0, 1)
        w_h = self.conv_1(w_h)
        w_h = torch.transpose(w_h, 0, 1)

        ### GM_V ###
        a = 0
        w_v = []
        for i in range(an):
            img = warped_h[:, i:i + 1, :, :, :]
            img = img.view(1, an, 6, h, w)
            warped = self.hang_conv(img)
            w_v.append(warped)
            a = a + 1
        w_v = torch.stack(w_v, dim=0)
        w_v = w_v.view(1, an2, 6, h, w)
        w_v = torch.transpose(w_v, 1, 2)
        w_v = self.residual_learning_1(w_v)
        w_v = w_v.view(6, an2, h, w)
        w_v = torch.transpose(w_v, 0, 1)
        w_v = self.conv_1(w_v)
        w_v = torch.transpose(w_v, 0, 1)

        ### Sum ###
        warped_fuse = warped_img[:, :, 0, :, :] + warped_img[:, :, 1, :, :] + warped_img[:, :, 2, :, :] + warped_img[:, :, 3, :, :]+ warped_img[:, :, 4, :, :]+ warped_img[:, :, 5, :, :]
        lf = w_h + w_v + warped_fuse

        return lf

class ResBlock(nn.Module):
    def __init__(self, an2, kernel_size, i):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))

        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out

        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)


def make_ResBlock(layer_num, an2, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock(an2, kernel_size, i))
    return nn.Sequential(*layers)

class FuseBlock(nn.Module):
    def __init__(self, a, kernel_size, i):
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(a, a, kernel_size=kernel_size, stride=(1, 1), padding=(1, 1))

        self._initialize_weights(a, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out

        return out

    def _initialize_weights(self, a, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * a * kernel_size[0] * kernel_size[1]))
        else:
            sigma = math.sqrt(2 / (a * a * kernel_size[0] * kernel_size[1]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)


def make_FuseBlock(layer_num, a, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(FuseBlock(a, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_1(nn.Module):
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_1, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))

        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out

        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)

def make_ResBlock_1(layer_num, an2, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_1(an2, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_2(nn.Module):
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_2, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))

        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out

        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)


def make_ResBlock_2(layer_num, an2, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_2(an2, kernel_size, i))
    return nn.Sequential(*layers)

class ResBlock_3(nn.Module):
    def __init__(self, an2, kernel_size, i):
        super(ResBlock_3, self).__init__()
        self.conv1 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(an2, an2, kernel_size=kernel_size, stride=(1, 1, 1), padding=(1, 1, 1))

        self._initialize_weights(an2, kernel_size, i)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = x + out

        return out

    def _initialize_weights(self, an2, kernel_size, i):
        if i == 0:
            sigma = math.sqrt(2 / (1 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        else:
            sigma = math.sqrt(2 / (an2 * an2 * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.normal_(std=sigma)
                m.bias.data.fill_(0)


def make_ResBlock_3(layer_num, an2, kernel_size):
    layers = []
    for i in range(layer_num):
        layers.append(ResBlock_3(an2, kernel_size, i))
    return nn.Sequential(*layers)


