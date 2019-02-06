from __future__ import print_function
import argparse
import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetCls

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--num_points', type=int, default=1000, help='input batch size')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')


opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root ='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/data_final/roofn3d_data_damage_all/test',
                           task='classification',
                           mode = 'test',
                           npoints = opt.num_points,
                           min_pts=0,
                           load_in_memory=True,
                           num_seg_class=5)
testdataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=False,
                                             num_workers=opt.workers)
num_batch = len(test_dataset)/opt.batchSize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classifier = PointNetCls(k = len(test_dataset.classes)).to(device)

opt.model = '/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/models/final_damage/cls/cls_model_49.pth'
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_test_correct = 0
n_log = 100

"""
for i, data in enumerate(testdataloader, 0):

    points, target = data
    points, target = points.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
    points = points.transpose(2, 1)
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    total_test_correct += correct.item()

    if i % n_log == 0:
        print('[%d/%d] accuracy: %f' % (i, num_batch, correct.item()/float(opt.batchSize)))"""

total_points = 0
for i, data in enumerate(test_dataset):
    point, target = data
    point = point.view(1, point.size(0), point.size(1))
    target = target.view(1, target.size(0))
    point, target = point.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
    point = point.transpose(2, 1)
    pred, _ = classifier(point)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    total_test_correct += correct.item()
    if i % n_log == 0:
        print('processing: ', i)


test_acc = float(total_test_correct) / float(len(test_dataset))
print('test_accuracy: {}'.format(test_acc))
