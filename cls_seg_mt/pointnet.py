from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from utils.util_layers import Dense


AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


class PointCNN(nn.Module):

    def __init__(self, num_classes):
        super(PointCNN, self).__init__()

        self.num_classes = num_classes
        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, self.num_classes, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, multi_task=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.multi_task = multi_task
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.multi_task:
            return x, torch.cat([x.view(-1, 1024, 1).repeat(1, 1, n_pts), pointfeat], 1), trans
        else:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
                return torch.cat([x, pointfeat], 1), trans


class PointNetMultiTask(nn.Module):
    def __init__(self, cls_k = 2, seg_k = 2):
        super(PointNetMultiTask, self).__init__()
        self.cls_k = cls_k
        self.seg_k = seg_k
        self.feat = PointNetfeat(multi_task=True)
        self.cls_fc1 = nn.Linear(1024, 512)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_fc3 = nn.Linear(256, cls_k)
        self.cls_bn1 = nn.BatchNorm1d(512)
        self.cls_bn2 = nn.BatchNorm1d(256)
        self.cls_relu = nn.ReLU()
        self.seg_conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_conv4 = torch.nn.Conv1d(128, self.seg_k, 1)
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

    def forward(self, x):

        # shared feature extractor
        x_cls, x_seg, trans = self.feat(x)

        # segmentation
        x_seg = F.relu(self.seg_bn1(self.seg_conv1(x_seg)))
        x_seg = F.relu(self.seg_bn2(self.seg_conv2(x_seg)))
        x_seg = F.relu(self.seg_bn3(self.seg_conv3(x_seg)))
        x_seg = self.seg_conv4(x_seg)
        x_seg = x_seg.transpose(2, 1).contiguous()
        seg_logsoft = F.log_softmax(x_seg, dim=-1)

        # classification
        x_cls = F.relu(self.cls_bn1(self.cls_fc1(x_cls)))
        x_cls = F.relu(self.cls_bn2(self.cls_fc2(x_cls)))
        x_cls = self.cls_fc3(x_cls)
        cls_logsoft =  F.log_softmax(x_cls, dim=-1)

        return cls_logsoft, seg_logsoft, trans


class PointNetCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.3)
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x, dim=-1)
        return x, trans


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
