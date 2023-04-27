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


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding2D, self).__init__()

        self.channels = int(np.ceil(channels/2))
        inv_freq = 1. / (10000
                         **(torch.arange(0, self.channels, 2).float()/self.channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x:Tensor):
        '''
        :param x:Tensor: A 4d tensor of the size (batchsize, w, h, ch)
        :return: Positional Encoding Matrix of size (batchsize, w, h, ch)
        '''
        if len(x.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d (batchsize, w, h, ch)")
        batch_size, w, h, ori_ch = x.shape
        #print(x.shape)
        pos_x = torch.arange(w, device=x.device).type(self.inv_freq.type())
        pos_y = torch.arange(h, device=x.device).type(self.inv_freq.type())
        '''
        pos_x = [0,1,2,3,...,w] shape (w,)
        pos_y = [0,1,2,3,...,h] shape (h,)
        '''
        #print("pos ",pos_x.shape,pos_y.shape)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        '''
        sin_inp_x.shape = (w,channels/4)
        sin_inp_y.shape = (h,channels/4)
        '''
        #print(sin_inp_x.shape, sin_inp_y.shape)
        emb_x = torch.cat((sin_inp_x.sin(),sin_inp_x.cos()),dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(),sin_inp_y.cos()),dim=-1)
        #print("emb_xory ",emb_x.shape, emb_y.shape)
        emb = torch.zeros((w,h,self.channels*2),device=x.device).type(x.type())
        #print("emb ",emb.shape)
        emb[:,:,:self.channels] = emb_x
        emb[:,:, self.channels:2*self.channels] = emb_y


        return emb[None,:,:,:ori_ch].repeat(batch_size,1,1,1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        print(channel)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x:Tensor):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) /
                         math.sqrt(c))  #[b, h*w, h*w]
        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()
        # Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        V = self.value(S1)
        # Ype = self.positional_encoding_2d(Yc, Yh, Yw)
        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)
        Q = self.query(Y1)
        K = self.key(Y1)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)
        Z = self.conv(x)
        Z = Z * S
        Z = torch.cat([Z, Y2], dim=1)
        return Z

class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        self.conv = nn.Sequential(
            nn.Conv2d(Ychannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(Schannels,
                      Schannels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True))

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class U_Transformer(nn.Module):
    def __init__(self, in_channels, classes, bilinear=True):
        super(U_Transformer, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.MHSA = MultiHeadSelfAttention(512)
        self.up1 = TransformerUp(512, 256)
        self.up2 = TransformerUp(256, 128)
        self.up3 = TransformerUp(128, 64)

        self.outc = nn.Conv2d(64, classes, kernel_size=1)
        self.active = torch.nn.Sigmoid()
        #self. = OutConv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        logits = self.active(logits)
        return logits