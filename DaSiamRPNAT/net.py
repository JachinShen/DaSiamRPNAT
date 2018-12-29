# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Modified by Jachin Shen (jachinshen@foxmail.com)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SiamRPNBIG(nn.Module):
    def __init__(self, feat_in=512, feature_out=512, anchor=5):
        super(SiamRPNBIG, self).__init__()
        self.anchor = anchor
        self.feature_out = feature_out
        self.featureExtract = nn.Sequential(
            nn.Conv2d(3, 192, 11, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(512, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 3),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 3),
            nn.BatchNorm2d(512),
        )
        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        self.r1_kernel = []
        self.cls1_kernel = []
        
        self.made_at = False
        
    def forward(self, x):
        x_f = self.featureExtract(x)
        score = F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)
        
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
                score
               #F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def temple(self, z):
        z_f = self.featureExtract(z)
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
    
    def make_at(self, x, label):
        x_f = self.featureExtract(x)
        cls1_kernel = Variable(self.cls1_kernel.data, requires_grad=True)
        self.optimizer = torch.optim.Adam([
            {"params":self.conv_cls2.parameters()},
            {"params":cls1_kernel}
        ], lr = 0.000001)
        
        for i in range(10):
            score = F.conv2d(self.conv_cls2(x_f), cls1_kernel)
            score = F.softmax(score)
            at = label * score
            self.zero_grad()
            loss = torch.nn.MSELoss()(score, at)
            print(loss)
            loss.backward(retain_graph=True)
            self.optimizer.step()
         
        self.cls1_kernel = cls1_kernel
    def make_at_small(self, x, label):
        x_f = self.featureExtract(x)
        cls1_kernel = Variable(self.cls1_kernel.data, requires_grad=True)
        self.optimizer = torch.optim.Adam([
            {"params":self.conv_cls2.parameters()},
            {"params":cls1_kernel}
        ], lr = 0.000001)
        
        for i in range(1):
            score = F.conv2d(self.conv_cls2(x_f), cls1_kernel)
            score = F.softmax(score)
            at = label * score
            self.zero_grad()
            loss = torch.nn.MSELoss()(score, at)
            print(loss)
            loss.backward(retain_graph=True)
            self.optimizer.step()
         
        self.cls1_kernel = cls1_kernel
