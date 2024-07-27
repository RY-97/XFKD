import numpy as np
import torch
import math

from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax



class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        b, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(b, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(b, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(b, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(b, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        b, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(b, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(b, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(b, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(b, C, height, width)

        out = self.gamma * out + x
        return out


class DAN(nn.Module):
    def __init__(self, in_channels):
        super(DAN, self).__init__()
        self.sa = PAM_Module(in_channels)  # 空间注意力模块
        self.sc = CAM_Module(in_channels)  # 通道注意力模块

    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        feat = x
        sa_feat = self.sa(feat)
        sc_feat = self.sc(feat)


        feat_sum = sa_feat + sc_feat  # 两个注意力模块结果相加

        output = feat_sum

        return output


# if __name__ == '__main__':
#     x = torch.randn(1, 64, 32, 48)
#     b, c, h, w = x.shape
#     net = DAN(in_channels=c)
#     y = net(x)