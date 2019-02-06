from __future__ import print_function
import argparse
import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetMultiTask
import os

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='data/test', help='path to input data')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--num_points', type=int, default=1000, help='input batch size')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')


opt = parser.parse_args()

test_dataset = PartDataset(root=os.path.join(opt.input_path, 'test'),
                           task='multi_task',
                           mode = 'test',
                           npoints = opt.num_points,
                           min_pts=0,
                           load_in_memory=True,
                           num_seg_class=5)
testdataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=opt.workers)
num_batch = len(test_dataset)/opt.batchSize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(test_dataset.classes)
num_seg_classes = test_dataset.num_seg_classes

classifier = PointNetMultiTask(cls_k=num_classes, seg_k=num_seg_classes).to(device)

classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

n_log = 100

total_points = 0
total_cls_test_correct = 0
total_seg_test_correct = 0
for i, data in enumerate(test_dataset):

    points, target_cls, target_seg = data
    points = points.view(1, points.size(0), points.size(1))
    target_cls = target_cls.view(1, target_cls.size(0))
    target_seg = target_seg.view(1, target_seg.size(0))
    points, target_cls, target_seg = points.to(device, non_blocking=True), \
                                     target_cls[:, 0].to(device, non_blocking=True), \
                                     target_seg.to(device, non_blocking=True)

    points = points.transpose(2, 1)
    pred_cls, preg_seg, _ = classifier(points)
    preg_seg = preg_seg.view(-1, num_seg_classes)
    target_seg = target_seg.view(-1, 1)[:, 0] - 1

    cls_pred_choice = pred_cls.data.max(1)[1]
    correct_cls = cls_pred_choice.eq(target_cls.long().data).cpu().sum()
    total_cls_test_correct += correct_cls.item()
    seg_pred_choice = preg_seg.data.max(1)[1]
    correct_seg = seg_pred_choice.eq(target_seg.long().data).cpu().sum()
    total_seg_test_correct += correct_seg.item()
    total_points += target_seg.size(0)

    if i % n_log == 0:
        print('processing: {}/{}'.format(i, len(test_dataset)))


test_cls_acc = float(total_cls_test_correct) / float(len(test_dataset))
test_seg_acc = float(total_seg_test_correct) / float(total_points)
print('test_cls_accuracy: {}'.format(test_cls_acc))
print('test_seg_accuracy: {}'.format(test_seg_acc))
