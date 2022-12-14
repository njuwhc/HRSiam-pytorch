from __future__ import absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.getcwd())

import glob
from itertools import islice
from siamrpn import HRSiamRPNTracker
import numpy as np


if __name__=='__main__':


    #seq_dir = os.path.expanduser('D:\Dataset\OTB100\Diving\\')  # seq_dir='F:\\data\\OTB100\\Crossing\img'
    seq_dir = 'otb/Walking/'
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))  # img_files是视频序列 glob.glob获取指定目录下的所有jpg文件,再进行排序
    # img_files[0]='F:\\data\\OTB100\\Crossing\\img\\0001.jpg'  依次类推
    # img_files = sorted(glob.glob('F:\data\OTB100\Crossing\img/*.jpg'))  #这个跟上面一样
    
    with open(seq_dir + 'groundtruth_rect.txt') as f:
        if '\t' in f.readline():
           anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')  # anno[0]=array([205.,151.,17.,50.])就是第一帧中的groundtruth,依此类推
        elif ',' in f.readline():
           anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt',delimiter=',')  # anno[0]=array([205.,151.,17.,50.])就是第一帧中的groundtruth,依此类推


    net_path = 'siamrpn_otb_1.pth'
    tracker = HRSiamRPNTracker(net_path)
    tracker.track(img_files, anno[0], visualize=True)
