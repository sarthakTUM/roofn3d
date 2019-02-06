from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import LambdaLR
from datasets import PartDataset
from pointnet import  PointNetMultiTask
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json


def plot_graphs(graph_dict, save_dict=True):
    # train and test loss curves
    plt.plot(graph_dict['epochs'], graph_dict['train_loss'], label='train_loss')
    plt.plot(graph_dict['epochs'], graph_dict['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()

    # train and test classification accuracies
    plt.plot(graph_dict['epochs'], graph_dict['train_cls_acc'], label='train_cls_acc')
    plt.plot(graph_dict['epochs'], graph_dict['val_cls_acc'], label='val_cls_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Classification Accuracy')
    plt.title('Training and Validation Classification Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'cls_accuracy.png'))
    plt.close()

    # train and test classification loss
    plt.plot(graph_dict['epochs'], graph_dict['train_cls_loss'], label='train_cls_loss')
    plt.plot(graph_dict['epochs'], graph_dict['val_cls_loss'], label='val_cls_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Classification Loss')
    plt.title('Training and Validation Classification Loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'cls_loss.png'))
    plt.close()

    # train and test segmentation accuracies
    plt.plot(graph_dict['epochs'], graph_dict['train_seg_acc'], label='train_seg_acc')
    plt.plot(graph_dict['epochs'], graph_dict['val_seg_acc'], label='val_seg_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation Accuracy')
    plt.title('Training and Validation Segmentation Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'seg_accuracy.png'))
    plt.close()

    # train and test segmentation loss
    plt.plot(graph_dict['epochs'], graph_dict['train_seg_loss'], label='train_seg_loss')
    plt.plot(graph_dict['epochs'], graph_dict['val_seg_loss'], label='val_seg_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Segmentation loss')
    plt.title('Training and Validation Segmentation loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'seg_loss.png'))
    plt.close()

    if save_dict:
        j = json.dumps(graph_dict)
        f = open(os.path.join(plots_dir, "stats.json"), "w")
        f.write(j)
        f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/train', help='path to input data')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--num_points', type=int, default=1000, help='input batch size')
parser.add_argument('--min_points', type=int, default=0, help='smallest point cloud')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='models/mt',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--lr_decay_rate', type=float, default = 0.05,  help='model path')

opt = parser.parse_args()
print (opt)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device: {}'.format(device))

model_dir = opt.outf

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


dataset = PartDataset(root = os.path.join(opt.input_path, 'train'),
                      task='multi_task',
                      npoints = opt.num_points,
                      min_pts=0,
                      load_in_memory=True,
                      num_seg_class=5)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batchSize,
                                         shuffle=True,
                                         num_workers=int(opt.workers))

val_dataset = PartDataset(root = os.path.join(opt.input_path, 'val'),
                          task='multi_task',
                          mode = 'val',
                          npoints = opt.num_points,
                          min_pts = 0,
                          load_in_memory=True,
                          num_seg_class=5)
valdataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=opt.batchSize,
                                            shuffle=True,
                                            num_workers=int(opt.workers))

print('train: {} test: {}'.format(len(dataset), len(val_dataset)))
num_classes = len(dataset.classes)
num_seg_classes = dataset.num_seg_classes
print('classification classes', num_classes)
print('segmentation classes', num_seg_classes)

classifier = PointNetMultiTask(cls_k=num_classes, seg_k=num_seg_classes).to(device)

start_epoch=-1

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))
    # TODO update start_epoch from pre-trained


optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, classifier.parameters()),
                      lr=0.01,
                      momentum=0.9)
lambda_lr = lambda epoch: 1 / (1 + (opt.lr_decay_rate * epoch))
lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr, last_epoch=start_epoch)

num_train_batch = len(dataset)/opt.batchSize
num_test_batch = len(val_dataset)/opt.batchSize
n_log = 100

epochs = []
train_cls_acc = []
train_cls_loss = []
train_seg_acc = []
train_seg_loss = []
train_loss = []
test_cls_acc = []
test_seg_acc = []
test_loss = []
test_cls_loss = []
test_seg_loss = []


plots_dir = os.path.join(model_dir, 'plots')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

for epoch in range(opt.nepoch):
    epoch_train_loss = 0
    epoch_test_loss = 0
    train_batch_num = 0
    test_batch_num = 0

    total_train_cls_correct = 0
    total_train_seg_correct = 0
    total_train_loss = 0
    total_train_cls_loss = 0
    total_train_seg_loss = 0
    classifier.train()
    for i, data in enumerate(dataloader, 0):
        train_batch_num += 1
        points, target_cls, target_seg = data
        points, target_cls, target_seg = points.to(device, non_blocking=True), \
                                         target_cls[:, 0].to(device, non_blocking=True), \
                                         target_seg.to(device, non_blocking=True)
        points = points.transpose(2,1)
        optimizer.zero_grad()
        pred_cls, preg_seg, _ = classifier(points)
        preg_seg = preg_seg.view(-1, num_seg_classes)
        target_seg = target_seg.view(-1, 1)[:, 0] - 1

        loss_cls = F.nll_loss(pred_cls, target_cls.long())
        loss_seg = F.nll_loss(preg_seg, target_seg.long())

        total_loss = loss_cls + loss_seg

        epoch_train_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

        cls_pred_choice = pred_cls.data.max(1)[1]
        correct_cls = cls_pred_choice.eq(target_cls.long().data).cpu().sum()
        total_train_cls_correct += correct_cls.item()
        seg_pred_choice = preg_seg.data.max(1)[1]
        correct_seg = seg_pred_choice.eq(target_seg.long().data).cpu().sum()
        total_train_seg_correct += correct_seg.item()
        total_train_loss += total_loss.item()
        total_train_cls_loss += loss_cls.item()
        total_train_seg_loss += loss_seg.item()

        if i%n_log == 0:
            print('{}:{}/{}, train loss: {}, classification_accuracy: {}, segmentation_accuracy: {}'.format(epoch,
                                                                                                            i,
                                                                                                            int(num_train_batch),
                                                                                                            total_loss.item(),
                                                                                                            correct_cls.item()/float(opt.batchSize),
                                                                                                            correct_seg.item()/float(opt.batchSize * opt.num_points)))


    epoch_train_cls_acc = float(total_train_cls_correct) / float(len(dataset))
    epoch_train_seg_acc = float(total_train_seg_correct) / float(len(dataset)*opt.num_points)
    epoch_train_loss = float(total_train_loss) / float(num_train_batch)
    epoch_train_cls_loss = float(total_train_cls_loss) / float(num_train_batch)
    epoch_train_seg_loss = float(total_train_seg_loss)/float(num_train_batch)
    print('train_cls_accuracy: {}'.format(epoch_train_cls_acc))
    print('train_seg_accuracy: {}'.format(epoch_train_seg_acc))
    print('train_loss: {}'.format(epoch_train_loss))

    total_test_cls_correct = 0
    total_test_seg_correct = 0
    total_test_loss = 0 # this is not used because the epoch_test_loss is updated inside the batch loop as a running sum
    total_test_cls_loss = 0
    total_test_seg_loss = 0
    classifier.eval()
    for j, data in enumerate(valdataloader, 0):
        test_batch_num += 1
        points, target_cls, target_seg = data
        points, target_cls, target_seg = points.to(device, non_blocking=True), \
                                         target_cls[:, 0].to(device, non_blocking=True), \
                                         target_seg.to(device, non_blocking=True)
        points = points.transpose(2, 1)
        pred_cls, preg_seg, _ = classifier(points)
        preg_seg = preg_seg.view(-1, num_seg_classes)
        target_seg = target_seg.view(-1, 1)[:, 0] - 1

        loss_cls = F.nll_loss(pred_cls, target_cls.long())
        loss_seg = F.nll_loss(preg_seg, target_seg.long())

        total_loss = loss_cls + loss_seg

        epoch_test_loss += total_loss.item()

        cls_pred_choice = pred_cls.data.max(1)[1]
        correct_cls = cls_pred_choice.eq(target_cls.long().data).cpu().sum()
        total_test_cls_correct += correct_cls.item()
        seg_pred_choice = preg_seg.data.max(1)[1]
        correct_seg = seg_pred_choice.eq(target_seg.long().data).cpu().sum()
        total_test_seg_correct += correct_seg.item()
        total_test_cls_loss += loss_cls.item()
        total_test_seg_loss += loss_seg.item()

    epoch_test_cls_acc = float(total_test_cls_correct) / float(len(val_dataset))
    epoch_test_seg_acc = float(total_test_seg_correct) / float(len(val_dataset)*opt.num_points)
    epoch_test_loss = float(epoch_test_loss) / float(num_test_batch)
    epoch_test_cls_loss = float(total_test_cls_loss) / float(num_test_batch)
    epoch_test_seg_loss = float(total_test_seg_loss)/float(num_test_batch)
    print('val_cls_accuracy: {}'.format(epoch_test_cls_acc))
    print('val_seg_accuracy: {}'.format(epoch_test_seg_acc))
    print('val_loss: {}'.format(epoch_test_loss))

    epochs.append(epoch)
    train_cls_acc.append(epoch_train_cls_acc)
    train_cls_loss.append(epoch_train_cls_loss)
    train_seg_acc.append(epoch_train_seg_acc)
    train_seg_loss.append(epoch_train_seg_loss)
    train_loss.append(epoch_train_loss)
    test_cls_acc.append(epoch_test_cls_acc)
    test_seg_acc.append(epoch_test_seg_acc)
    test_loss.append(epoch_test_loss)
    test_cls_loss.append(epoch_test_cls_loss)
    test_seg_loss.append(epoch_test_seg_loss)
    plot_graphs({'epochs':epochs,
                 'train_cls_acc':train_cls_acc,
                 'train_cls_loss':train_cls_loss,
                 'train_seg_acc':train_seg_acc,
                 'train_seg_loss':train_seg_loss,
                 'train_loss':train_loss,
                 'val_cls_acc':test_cls_acc,
                 'val_seg_acc':test_seg_acc,
                 'val_loss':test_loss,
                 'val_cls_loss':test_cls_loss,
                 'val_seg_loss':test_seg_loss})
    torch.save(classifier.state_dict(), '%s/mt_model_%d.pth' % (model_dir, epoch))
