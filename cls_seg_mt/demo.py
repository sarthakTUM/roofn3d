from __future__ import print_function
import argparse
import numpy as np
from datasets import PartDataset
from pointnet import PointNetDenseCls, PointNetCls, PointNetMultiTask
import matplotlib.pyplot as plt
import torch
import random

parser = argparse.ArgumentParser()

parser.add_argument('--seg_model', type=str, default = 'models/segmentation_damaged.pth',  help='segmentation model path for damaged dataset')
parser.add_argument('--cls_model', type=str, default = 'models/classification_damaged.pth',  help='classification model path for damaged dataset')
parser.add_argument('--mt_model', type=str, default = 'models/multitask_damaged.pth',  help='Multi-Task model path for damaged dataset')
parser.add_argument('--seg_all_model', type=str, default = 'models/segmentation_complete.pth',  help='segmentation model path for non-damaged dataset')
parser.add_argument('--cls_all_model', type=str, default = 'models/classification_complete.pth',  help='classification model path for non-damaged dataset')
parser.add_argument('--mt_all_model', type=str, default = 'models/multitask_complete.pth',  help='Multi-Task model path for non-damaged dataset')
parser.add_argument('--idx', type=int, default = 10,   help='model index')
parser.add_argument('--num_points', type=int, default=1000, help='input batch size')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')

opt = parser.parse_args()

def get_damaged_points(pts):
    pts_kept = []
    circle_radius = random.randint(1, 3)
    i = np.random.choice(len(pts))
    random_pt = pts[i]
    new_pts = []
    for pt_idx, point in enumerate(pts):
        if np.linalg.norm(point-random_pt) > circle_radius:
            new_pts.append(point)
            pts_kept.append(pt_idx)
    return np.array(new_pts), pts_kept

dataset = PartDataset(root = 'demo_data',
                      task='multi_task',
                      npoints = opt.num_points,
                      mode='test',
                      min_pts=0,
                      num_seg_class=5,
                      load_in_memory=False)

cls_class_to_idx = dataset.classes
cls_idx_to_class = {v:k for k,v in dataset.classes.items()}

idx = opt.idx

print("model %d/%d" %( idx, len(dataset)))

point, cls, seg = dataset[idx]
damaged_point, pt_idx = get_damaged_points(point.numpy())
damaged_point = torch.from_numpy(damaged_point)
damaged_seg = seg[pt_idx]

original_point = point.numpy()
damaged_point_np = damaged_point.numpy()
print(point.size(), seg.size())

print('loading segmentation network for damaged data')
seg_classifier = PointNetDenseCls(k = dataset.num_seg_classes)
seg_classifier.load_state_dict(torch.load(opt.seg_model))

print('loading classification network for damaged data')
cls_classifier = PointNetCls(k=len(dataset.classes))
cls_classifier.load_state_dict(torch.load(opt.cls_model))

print('loading multi-task network for damaged data')
mt_classifier = PointNetMultiTask(cls_k=len(dataset.classes), seg_k=dataset.num_seg_classes)
mt_classifier.load_state_dict(torch.load(opt.mt_model))

print('loading segmentation network for non-damaged data')
seg_classifier_all = PointNetDenseCls(k = dataset.num_seg_classes)
seg_classifier_all.load_state_dict(torch.load(opt.seg_all_model))

print('loading classification network for non-damaged data')
cls_classifier_all = PointNetCls(k=len(dataset.classes))
cls_classifier_all.load_state_dict(torch.load(opt.cls_all_model))

print('loading multi-task network for non-damaged data')
mt_classifier_all = PointNetMultiTask(cls_k=len(dataset.classes), seg_k=dataset.num_seg_classes)
mt_classifier_all.load_state_dict(torch.load(opt.mt_all_model))

seg_classifier.eval()
cls_classifier.eval()
mt_classifier.eval()
seg_classifier_all.eval()
cls_classifier_all.eval()
mt_classifier_all.eval()


point = point.transpose(1,0).contiguous()
point = point.view(1, point.size()[0], point.size()[1])
pred_cls_all, _ = cls_classifier_all(point)
pred_cls_all_choice = pred_cls_all.data.max(1)[1]
pred_seg_all, _ = seg_classifier_all(point)
pred_seg_all_choice = pred_seg_all.data.max(2)[1]
pred_cls_all_mt, preg_seg_all_mt, _ = mt_classifier_all(point)
pred_cls_mt_all_choice = pred_cls_all_mt.data.max(1)[1]
pred_seg_mt_all_choice = preg_seg_all_mt.data.max(2)[1]

damaged_point = damaged_point.transpose(1,0).contiguous()
damaged_point = damaged_point.view(1, damaged_point.size()[0], damaged_point.size()[1])
pred_cls, _ = cls_classifier(damaged_point)
pred_cls_choice = pred_cls.data.max(1)[1]
pred_seg, _ = seg_classifier(damaged_point)
pred_seg_choice = pred_seg.data.max(2)[1]
pred_cls_mt, preg_seg_mt, _ = mt_classifier(damaged_point)
pred_cls_mt_choice = pred_cls_mt.data.max(1)[1]
pred_seg_mt_choice = preg_seg_mt.data.max(2)[1]

"""
print('ground-truth class for non-damaged data: ', cls_idx_to_class[cls.item()])
print('predicted class by single-task network for non-damaged data: ', cls_idx_to_class[pred_cls_all_choice.item()])
print('predicted class multi-task network for non-damaged data: ', cls_idx_to_class[pred_cls_mt_all_choice.item()])
print('ground-truth segmentation for non-damaged data: ', seg.view(-1,1)[:,0] - 1, seg.shape)
print('predicted segmentation by single-task network for non-damaged data: ', pred_seg_all_choice.squeeze(0), pred_seg_all_choice.shape)
print('predicted segmentation by multi-task network for non-damaged data: ', pred_seg_mt_all_choice.squeeze(0), pred_seg_mt_all_choice.shape)
print('ground-truth segmentation for damaged data: ', damaged_seg.view(-1,1)[:,0] - 1, seg.shape)
print('predicted segmentation by single-task network for damaged data: ', pred_seg_choice.squeeze(0), pred_seg_choice.shape)
print('predicted segmentation by multi-task network for damaged data: ', pred_seg_mt_choice.squeeze(0), pred_seg_mt_choice.shape)
"""

del seg_classifier
torch.cuda.empty_cache()
del cls_classifier
torch.cuda.empty_cache()
del mt_classifier
torch.cuda.empty_cache()

# VIZUALIZATION
cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg.numpy().astype(int) - 1, :]
pred_color_st_nondamage = cmap[pred_seg_all_choice.numpy()[0], :]
pred_color_mt_nondamage = cmap[pred_seg_mt_all_choice.numpy()[0], :]
damaged_gt = cmap[damaged_seg.numpy().astype(int) - 1, :]
pred_color_st_damaged = cmap[pred_seg_choice.numpy()[0], :]
pred_color_mt_damaged = cmap[pred_seg_mt_choice.numpy()[0], :]


import pptk

print('showing results. Press ENTER for continuing...')

# ground-truth
print('Ground truth non-damaged segmentation')
v1 = pptk.viewer(original_point)
v1.attributes(gt)
v1.set(point_size=0.1)
v1.wait()

# single-task segmentation
print('Predicted segmentation by single-task network for non-damaged data')
v2 = pptk.viewer(original_point)
v2.attributes(pred_color_st_nondamage)
v2.set(point_size=0.1)
v2.wait()

# multi-task segmentation
print('Predicted segmentation by multi-task network for non-damaged data')
v3 = pptk.viewer(original_point)
v3.attributes(pred_color_mt_nondamage)
v3.set(point_size=0.1)
v3.wait()


# damaged ground-truth
print('Ground truth damaged segmentation')
v4 = pptk.viewer(damaged_point_np)
v4.attributes(damaged_gt)
v4.set(point_size=0.1)
v4.wait()


# damaged single-task segmentation
print('Predicted segmentation by single-task network for damaged data')
v5 = pptk.viewer(damaged_point_np)
v5.attributes(pred_color_st_damaged)
v5.set(point_size=0.1)
v5.wait()

# damaged multi-task segmentation
print('Predicted segmentation by multi-task network for damaged data')
v6 = pptk.viewer(damaged_point_np)
v6.attributes(pred_color_mt_damaged)
v6.set(point_size=0.1)
v6.wait()


viewers = [v1, v2, v3, v4, v5, v6]
for v in viewers:
    v.close()