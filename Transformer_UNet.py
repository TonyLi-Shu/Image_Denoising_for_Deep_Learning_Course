from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math
from torch import nn, Tensor
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0, max_len:int = 65538):
        super(PositionalEncoding,self).__init__()
        self. dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model, 2) * (-math.log(10000.0)/ d_model))
        pe = torch.zeros(max_len, 1 , d_model)
        pe[:, 0,  0::2] = torch.sin(position * div_term)
        pe[:, 0,  1::2] = torch.cos(position * div_term)
        pe = torch.permute(pe, (1, 0, 2))

        self.register_buffer('pe', pe)

    def forward(self, x:Tensor):
        '''
        :param x: (Batchsize, dim, h, w)
        :change it to (Batchsize, seq_len, dim)
        :return: (Batchsize, dim, h, w)
        '''
        #print(x.shape)
        n_bs, dim, H, W = x.size()
        x = torch.reshape(x, (n_bs, H * W, dim))
        #print(x.size())
        x = x + self.pe[:,x.size(1),:]
        x = torch.reshape(x, (n_bs, dim, H, W))
        return x

class Self_Attention(nn.Module):
    def __init__(self, d_model:int):
        super(Self_Attention, self).__init__()
        # (Batch_size, H*W, N_channels)
        self.d_model = d_model
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q: Tensor, K: Tensor, V:Tensor):
        #matmul = torch.matmul(Q, K.transpose(2, 1))
        Ktrans = K.transpose(2, 1)
        matmul = torch.einsum('bij,bjk->bik', Q,Ktrans)
        A = self.softmax( matmul/ math.sqrt(self.d_model))
        AV = torch.matmul(A, V)
        return AV

class Single_Self_Attention(nn.Module):
    def __init__(self, d_model: int):
        super(Single_Self_Attention, self).__init__()
        self.attention = Self_Attention(d_model)
        self.pos_Encoding = PositionalEncoding(d_model)
        self.fcQ = nn.Linear(d_model, d_model)
        self.fcK = nn.Linear(d_model, d_model)
        self.fcV = nn.Linear(d_model, d_model)
    def forward(self, x:Tensor):
        x = self.pos_Encoding(x)
        n_bs, d_channels, H, W = x.size()
        x = torch.reshape(x, (n_bs, H * W, d_channels))
        Q = self.fcQ(x)
        K = self.fcK(x)
        V = self.fcV(x)
        x = self.attention(Q, K, V)
        x = torch.reshape(x, (n_bs, d_channels, H, W))
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, d_model:int):
        super(Multi_Head_Self_Attention, self).__init__()

        self.d_model = d_model
        self.PositionalEncodingS = PositionalEncoding(d_model)
        self.PositionalEncodingY = PositionalEncoding(d_model*2)
        self.self_attention = Self_Attention(d_model)

        self.upperline = nn.Sequential(
            nn.Conv2d(d_model, int(d_model/4),1),
            nn.BatchNorm2d(int(d_model/4)),
            nn.ReLU()
        )


        self.bottomline1 = nn.Sequential(
            nn.Conv2d(2*d_model, d_model,1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.bottomline2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*d_model,2* d_model,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(2*d_model, d_model,1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )

        self.midconv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2)
        )

        self.fcQ = nn.Linear(d_model, d_model)
        self.fcK = nn.Linear(d_model, d_model)
        self.fcV = nn.Linear(d_model, d_model)
    def forward(self, s:Tensor, y:Tensor):
        #print("y.shape = ",y.shape)
        y = self.PositionalEncodingY(y)
        # print("y.shape = ", y.shape)
        qk = self.bottomline1(y)
        n_bs, d_channels, yH , yW = qk.size()
        qk = torch.reshape(qk, (n_bs, -1, self.d_model))
        Q = self.fcQ(qk)
        K = self.fcK(qk)

        s = self.PositionalEncodingS(s)
        v = self.upperline(s)
        n_bs, d_channels, sH, sW = v.size()
        v = torch.reshape(v, (n_bs, -1, self.d_model))
        V = self.fcV(v)

        #print(Q.size(), K.size(), V.size())
        mid = self.self_attention(Q,K,V)
        mid = torch.reshape(mid, (n_bs, self.d_model, yH, yW))
        #print(mid.size())
        mid = self.midconv(mid)
        #print(mid.size())
        mid = mid * s
        y = self.bottomline2(y)
        #print(y.size())
        out = torch.concat((mid,y),1)
        #print(out.size())
        return out

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Transformer_U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, out_ch=1):
        super(Transformer_U_Net, self).__init__()

        n1 = 4
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        #self.Conv5 = conv_block(filters[3], filters[4])

        self.mhsa = Single_Self_Attention(filters[3])

        #self.Up5 = up_conv(filters[4], filters[3])
        #self.Up5 = Multi_Head_Self_Attention(filters[3])
        #self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = Multi_Head_Self_Attention(filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = Multi_Head_Self_Attention(filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = Multi_Head_Self_Attention(filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5)

        d4 = self.mhsa(e4)
        #print("d4.shape = ", d4.shape)

        d3 = self.Up4(e3, d4)
        d3 = self.Up_conv4(d3)
        #print("d3.shape = ",d3.shape)

        d2 = self.Up3(e2,d3)
        d2 = self.Up_conv3(d2)

        d1 = self.Up2(e1,d2)
        d1 = self.Up_conv2(d1)
        #print("d1.shape = ",d1.shape)
        out = self.Conv(d1)
        out = self.active(out)


        return out



# TUNet = Transformer_U_Net(1,1)
# input = torch.rand((1,256,256)).repeat(4,1,1,1)
# print(input.size())
# out = TUNet(input)
# print(out.size())



