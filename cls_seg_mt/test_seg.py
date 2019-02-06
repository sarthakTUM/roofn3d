from __future__ import print_function
import argparse
import torch.utils.data
from datasets import PartDataset
from pointnet import PointNetDenseCls

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--num_points', type=int, default=1000, help='input batch size')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')


opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root='/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/data_final/roofn3d_data_damage_all/test',
                           task='segmentation',
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

classifier = PointNetDenseCls(k = test_dataset.num_seg_classes).to(device)

opt.model = '/media/sarthak/HDD/TUM/courses/sem_5/ADLCV/pointnet.pytorch/models/final_damage/seg/seg_model_30.pth'
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

total_test_correct = 0
n_log = 100

"""
for i, data in enumerate(testdataloader, 0):

    points, target = data
    points, target = points.to(device, non_blocking=True), target.to(device, non_blocking=True)
    points = points.transpose(2, 1)
    pred, _ = classifier(points)
    pred = pred.view(-1, test_dataset.num_seg_classes)
    target = target.view(-1, 1)[:, 0] - 1
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    total_test_correct += correct.item()

    if i % n_log == 0:
        print('[%d/%d] accuracy: %f' % (i, num_batch, correct.item() / float(opt.batchSize * opt.num_points)))"""

total_points = 0
for i, data in enumerate(test_dataset):
    point, target = data
    # print(point.shape, target.shape)
    point, target = point.to(device), target.to(device)
    point = point.view(1, point.size(0), point.size(1))
    target = target.view(1, target.size(0))
    point = point.transpose(2, 1)
    pred, _ = classifier(point)
    pred = pred.view(-1, test_dataset.num_seg_classes)
    target = target.view(-1, 1)[:, 0] - 1
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.long().data).cpu().sum()
    total_test_correct += correct.item()
    total_points += target.size(0)
    if i % n_log == 0:
            print('[%d/%d] accuracy: %f' % (i, len(test_dataset), correct.item()/float(target.size(0))))


# test_acc = float(total_test_correct) / float(len(test_dataset)*opt.num_points)
test_acc = float(total_test_correct) / float(total_points)
print('test_accuracy: {}'.format(test_acc))
