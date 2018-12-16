import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = '../dataset/'
seqlist_path = 'data/vot-otb.txt'
output_path = 'data/vot-otb.pkl'

# with open(seqlist_path,'r') as fp:
#     seq_list = fp.read().splitlines()
seq_list=[
    # "vot2013/cup",
    # "vot2013/iceskater",
    # "vot2013/juice"
    # "vot2014/ball",
    # "vot2014/bicycle",
    # "vot2014/drunk",
    # "vot2014/fish1",
    # "vot2014/hand1",
    # "vot2014/polarbear",
    # "vot2014/sphere",
    # "vot2014/sunshade",
    # "vot2014/surfing",
    # "vot2014/torus",
    # "vot2014/tunnel"
    "vot2015/bag",
    "vot2015/ball1",
    "vot2015/ball2",
    "vot2015/basketball",
    "vot2015/birds1",
    "vot2015/birds2",
    "vot2015/blanket",
    "vot2015/bmx",
    "vot2015/bolt1",
    "vot2015/bolt2",
    "vot2015/book",
    "vot2015/butterfly",
    "vot2015/car1",
    "vot2015/car2",
    "vot2015/crossing",
    "vot2015/dinosaur",
    "vot2015/fernando",
    "vot2015/fish1",
    "vot2015/fish2",
    "vot2015/fish3",
    "vot2015/fish4",
    "vot2015/girl",
    "vot2015/glove",
    "vot2015/godfather",
    "vot2015/graduate",
    "vot2015/gymnastics1",
    "vot2015/gymnastics2",
    "vot2015/gymnastics3",
    "vot2015/gymnastics4",
    "vot2015/hand",
    "vot2015/handball1",
    "vot2015/handball2",
    "vot2015/helicopter",
    "vot2015/iceskater1",
    "vot2015/iceskater2",
    "vot2015/leaves",
    "vot2015/marching",
    "vot2015/matrix",
    "vot2015/motocross1",
    "vot2015/motocross2",
    "vot2015/nature",
    "vot2015/octopus",
    "vot2015/pedestrian1",
    "vot2015/pedestrian2",
    "vot2015/rabbit",
    "vot2015/racing",
    "vot2015/road",
    "vot2015/shaking",
    "vot2015/sheep",
    "vot2015/singer1",
    "vot2015/singer2",
    "vot2015/singer3",
    "vot2015/soccer1",
    "vot2015/soccer2",
    "vot2015/soldier",
    "vot2015/sphere",
    "vot2015/tiger",
    "vot2015/traffic",
    "vot2015/tunnel",
    "vot2015/wiper"
         ]

data = {}
for i,seq in enumerate(seq_list):
    img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.jpg'])
    gt = np.loadtxt(seq_home+seq+'/groundtruth.txt',delimiter=',')

    print(seq, len(img_list), len(gt))
    assert len(img_list) == len(gt), "Lengths do not match!!"
    
    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seq] = {'images':img_list, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
