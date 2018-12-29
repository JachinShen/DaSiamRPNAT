# DaSiamIRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# Modified by Jachin Shen (jachinshen@foxmail.com)
# --------------------------------------------------------
#!/usr/bin/python

import sys
import cv2  # imread
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
from os.path import realpath, dirname, join

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, overlap_ratio, draw_rect

print("Start DaSiamRPNAT")

# load net
net_file = "../model/SiamRPNBIG.model"
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

img_home = '../dataset/OTB/'
data_path = '../data/vot-otb.pkl'
print("open vot-otb.pkl")
with open(data_path, 'rb') as fp:
    data = pickle.load(fp)

res_dict = {}
overlap = []
for k, (seqname, seq) in enumerate(data.items()):
    print("Processing " + seqname)
    rect_list = []
    img_list = seq['images']
    gt = seq['gt']
    img_dir = os.path.join(img_home, seqname)
    gt_init = gt[0]
    x, y, w, h = gt_init[0], gt_init[1], gt_init[2], gt_init[3]
    cx, cy = x + w//2, y + h // 2
    
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    image_file = os.path.join(img_dir, img_list[0])
    im = cv2.imread(image_file)  # HxWxC
    state = SiamRPN_init(im, target_pos, target_sz, net, gt[0])  # init tracker
    for i in range(len(img_list)):
        image_file = os.path.join(img_dir, img_list[i])
        if not image_file:
            break
        im = cv2.imread(image_file)  # HxWxC
        state = SiamRPN_track(state, im)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        rect_list.append(res.tolist())
        overlap.append(overlap_ratio(res, gt[i]))

    res_dict[seqname] = rect_list
    json.dump(res_dict, open("../results/DaSiamRPNAT11.json", 'w'), indent=2) 
