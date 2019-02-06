from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json


class PartDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints = 500,
                 min_pts = 500,
                 task='classification',
                 class_choice = None,
                 mode = 'train',
                 num_seg_class=None,
                 load_in_memory=False):
        self.min_pts = min_pts
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.task = task
        self.num_seg_class = num_seg_class
        self.pointsets = dict()
        self.segs = dict()
        self.load_in_memory = load_in_memory
        self.mode = mode

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        total_points = 0
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))

            # train validation split
            """
            choice = np.random.choice(len(a), int(len(a) * 0.9), replace=False)
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]"""

            for fn in fns:
                total_points += 1
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

        print('dataset contains {} pointclouds'.format(total_points))

        self.datapath = []
        pts_file = os.path.join(root, self.mode+'_pts.pkl')
        seg_file = os.path.join(root, self.mode+'_seg.pkl')
        # pts_file = os.path.join(root, 'train_pts.pkl') if self.mode in ['train'] else os.path.join(root, 'val_pts.pkl')
        # seg_file = os.path.join(root, 'train_seg.pkl') if self.mode in ['train'] else os.path.join(root, 'val_seg.pkl')
        files = [pts_file, seg_file]
        data_exists = all([os.path.exists(f) for f in files])
        if not data_exists:
            print('some files don\'t exist. Reading Dataset.')
        loaded = 0
        for item in self.cat:
            for fn in self.meta[item]:
                loaded += 1
                self.datapath.append((item, fn[0], fn[1]))
                if self.load_in_memory and not data_exists:
                    self.pointsets[fn[0]] = np.loadtxt(fn[0]).astype(np.float32)
                    self.segs[fn[1]] = np.loadtxt(fn[1]).astype(np.int64)
                    if loaded % 1000 == 0:
                        print('loaded {} points in memory'.format(loaded))
        if not data_exists:
            print('saving data in memory.')
            self.save_obj(self.pointsets, pts_file)
            self.save_obj(self.segs, seg_file)

        if self.load_in_memory and data_exists:
            print('load_in_memory is TRUE and data exists on file system')
            self.pointsets = self.load_obj(pts_file)
            self.segs = self.load_obj(seg_file)

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print('classification classes: ', self.classes)
        self.num_seg_classes = 0
        if self.task == 'segmentation' or self.task == 'multi_task':
            if not self.num_seg_class:
                for i in range(len(self.datapath)):
                    l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                    if l > self.num_seg_classes:
                        self.num_seg_classes = l
            else:
                self.num_seg_classes = self.num_seg_class

            print('segmentation classes: ', self.num_seg_classes)


    def __getitem__(self, index):

        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        if not self.load_in_memory:
            point_set = np.loadtxt(fn[1]).astype(np.float32)
            seg = np.loadtxt(fn[2]).astype(np.int64)
        else:
            point_set = self.pointsets[fn[1]]
            seg = self.segs[fn[2]]
            point_set = np.atleast_2d(point_set)
            seg = np.atleast_1d(seg)

        # sample with replacement if number of points in some pointclouds is less than npoints, else sample without replacement, taking all.
        if self.mode in ['train', 'val']:
            choice = np.random.choice(seg.size, self.npoints, replace=False) if self.min_pts >= self.npoints else np.random.choice(seg.size, self.npoints, replace=True)

            #resample
            point_set = point_set[choice, :]
            seg = seg[choice]

        point_set = torch.from_numpy(point_set).float()
        seg = torch.from_numpy(seg).float()

        assert len(seg) == len(point_set)

        cls = torch.from_numpy(np.array([cls])).float()
        if self.task == 'multi_task':
            return point_set, cls, seg
        elif self.task == 'classification':
            return point_set, cls
        elif self.task == 'segmentation':
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


    def load_obj(self, filename):
        import pickle
        with open(filename, "rb") as input_file:
            return pickle.load(input_file)

    def save_obj(self, obj, filename):
        import pickle
        with open(filename, "wb") as output_file:
            pickle.dump(obj, output_file)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
