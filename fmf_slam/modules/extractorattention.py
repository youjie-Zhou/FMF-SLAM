import torch
import torch.nn as nn
import torch.nn.functional as F
# from .mmtm import MMTM
import copy
import numpy as np
import math
from copy import deepcopy
from .FFTformer import Fuse,Fusecross
# from setrans import SETransConfig, SelfAttVisPosTrans
import time
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)
#
    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))
#
#

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class GateNetworkRNN(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(GateNetworkRNN, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        # 定义RNN层
        # self.rnn = nn.RNN(input_channels * 48 * 64, hidden_dim, batch_first=True)
        self.rnn = nn.RNN(input_channels, hidden_dim, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, feat2_sequence):
        # 将feat2序列展平成二维张量，以便输入到RNN
        feat2_flat = feat2_sequence.view(feat2_sequence.size(0),feat2_sequence.size(1), -1).permute(0,2,1)
        # print('1',feat2_sequence.shape,feat2_flat.shape)
        # 经过RNN层
        rnn_out, _ = self.rnn(feat2_flat)
        
        # 取RNN输出序列的最后一个时间步
        last_rnn_out = rnn_out[:, -1, :]
        
        # 经过全连接层
        weight = torch.sigmoid(self.fc(last_rnn_out)).permute(1,0)
        # print(weight.shape)
        return weight[0]
    

class CONV(nn.Module):
    def __init__(self, in_dim=1, norm_fn='batch'):
        super(CONV, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            b_norm = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            b_norm = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            b_norm = nn.InstanceNorm2d(64)
        else:
            b_norm = nn.Sequential()
        self.relu1 = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3),
            b_norm,
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        self.conv_rgb = CONV(in_dim=3, norm_fn=norm_fn)
        self.conv_dep = CONV(in_dim=3, norm_fn=norm_fn)
        self.rgbpool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.depthpool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.rgbpool2 = nn.MaxPool2d(kernel_size=(5, 5))
        self.depthpool2 = nn.MaxPool2d(kernel_size=(5, 5))
        self.depthpool3 = nn.MaxPool2d(kernel_size=(3, 3))
        self.depthpool4 = nn.MaxPool2d(kernel_size=(4, 4))
        self.in_planes = 64
        layer1 = self._make_layer(64, stride=1)
        layer2 = self._make_layer(96, stride=2)
        layer3 = self._make_layer(128, stride=2)
        # self.mmtmrgb =
        # self.mmtm0 = MMTM(64, 64, 4)
        # self.mmtm1 = MMTM(96, 96, 4)
        # self.mmtm2 = MMTM(128, 128, 4)
        # self.layerrgb1 = nn.Sequential(layer1, layer2)#, layer3)
        # self.layerdepth1 = copy.deepcopy(nn.Sequential(layer1, layer2))
        self.layerrgb1 = layer1#, layer3)
        self.layerdepth1 = copy.deepcopy(layer1)
        self.layerrgb2 = layer2
        self.layerdepth2 = copy.deepcopy(layer2)
        self.layerrgb3 = layer3
        self.layerdepth3 = copy.deepcopy(layer3)
        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)
        self.conv2d = nn.Conv2d(128, output_dim, kernel_size=1)
        self.dropout = None
        # self.fuse1 = Fuse(96,4)
        # self.fuse2 = Fuse(96,4)
        # self.fuse3 = Fusecross(96,4)
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        self.sattn1 = AttentionalPropagation(feature_dim=64, num_heads=4)
        self.sattn2 = AttentionalPropagation(feature_dim=64, num_heads=4)
        self.sattn3 = AttentionalPropagation(feature_dim=64, num_heads=4)
        # self.gate_net_rnn = GateNetworkRNN(output_dim, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def _make_layers(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        # x = x.view(b * n, c1, h1, w1)
        # print('extra',x[0].shape)
        # print('extra1',x[1].shape)
        # print(x.y)
        # print('2',x[0].shape,x[1].shape)
        # print('1',x[0].max(),x[0].mean(),x[1].max(),x[1].mean())#2.6400,1.8653, 2.5000,1.2864

        b1, n1, c1, h1, w1 = x[0].shape
        b2, n2, c2, h2, w2 = x[1].shape

        x[0] = x[0].view(b1*n1, c1, h1, w1)
        x[1] = x[1].view(b2*n2, c2, h2, w2)
        a = x[0]
        b = x[1]

        a = self.conv_rgb(a)
        b = self.conv_dep(b)

#######################################################
        apool = self.rgbpool1(a)
        bpool = self.depthpool1(b)
        apool = self.rgbpool2(apool)
        bpool = self.depthpool2(bpool)
        bs3,c3,w3,h3 = apool.size()
        bs4,c4,w4,h4 = bpool.size()
        apool = apool.view(bs3,c3,-1)
        bpool = bpool.view(bs4,c4,-1)
        bs1,c1,w1,h1 = a.size()
        bs2,c2,w2,h2 = b.size()
        a1 = a.view(bs1,c1,-1)
        b1 = b.view(bs2,c2,-1)
        a1s = self.sattn1(a1, apool)
        b1s = self.sattn2(b1, bpool)
        a1 = a1 + a1s
        b1 = b1 + b1s
        a = a1.view(bs1,c1,w1,h1)
        b = b1.view(bs2,c2,w2,h2)
#############################################################
#######################################################
        bpool = self.depthpool3(b)
        bpool = self.depthpool4(bpool)
        bs4,c4,w4,h4 = bpool.size()
        bpool = bpool.view(bs4, c4, -1)
        bs1, c1, w1, h1 = a.size()
        a1 = a.view(bs1, c1, -1)
        a1s = self.sattn3(a1, bpool)
        a1 = a1 + a1s
        a = a1.view(bs1, c1, w1, h1)
#############################################################
        a = self.layerrgb1(a)
        b = self.layerdepth1(b)
        # print('1',a.shape,b.shape)
        # a, b = self.mmtm0([a, b])
        a = self.layerrgb2(a)
        b = self.layerdepth2(b)
        # start_time = time.perf_counter()
        # a = self.fuse1(a,a)
        # b = self.fuse2(b,b)
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time  # 计算运行时间
        # print(f"self: {elapsed_time} 秒")

        # start_time = time.perf_counter()
        # a,b = self.fuse3(a,b)
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time  # 计算运行时间
        # print(f"cross: {elapsed_time} 秒")
        # print('2',a.shape,b.shape)
        # a, b = self.mmtm1([a, b])
        a = self.layerrgb3(a)
        b = self.layerdepth3(b)

        a = self.conv2(a)
        b = self.conv2d(b)
        # updated_b = torch.zeros_like(b)
        # weight = self.gate_net_rnn(b)
        # # print(weight)
        # for i in range(weight.size(0)):      # 遍历样本维度
        #     updated_b[i] = b[i] * weight[i]
        # print(b.shape)
        # print('b',b.shape)
        # x = a + b
        # print('5',a.shape,b.shape)
        # x = torch.cat([a, b], 1)
        # x = self.conv3(x)
        # x = x.unsqueeze(0)
        # print(x.shape)
        # print('2',a.shape,b.shape) 
        # print(a.max(),a.mean(),b.max(),b.mean())#3.0108,-0.0023,6.7079, 0.0155
        return a,b


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        # print('3')
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.mmtm0(x)
        x = self.layer2(x)
        x = self.mmtm1(x)
        x = self.layer3(x)
        x = self.mmtm2(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
