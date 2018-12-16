import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = '../dataset/OTB/'
seqlist_path = '../dataset/SEQUENCE'
output_path = '../data/vot-otb.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()
# seq_list=[
         # ]

data = {}
for i,seq in enumerate(seq_list):
    print(seq)
    img_list = sorted(["img/"+p for p in os.listdir(seq_home+seq+"/img") if os.path.splitext(p)[1] == '.jpg'])
    try:
        gt = np.loadtxt(seq_home+seq+'/groundtruth_rect.txt',delimiter=',')
    except:
        gt = np.loadtxt(seq_home+seq+'/groundtruth_rect.txt')
        

    if seq == "David":
        img_list = img_list[299:771]
    elif len(img_list) > len(gt):
        img_list = img_list[:len(gt)]

    print(seq, len(img_list), len(gt))
    assert len(img_list) == len(gt), "Lengths do not match!!"

    if i == len(seq_list)-1:
        print(gt.shape)
        print(gt)
    
    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seq] = {'images':img_list, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
