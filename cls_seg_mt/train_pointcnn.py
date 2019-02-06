from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from datasets import PartDataset
from pointnet import PointCNN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json



def plot_graphs(graph_dict, save_dict=True):
    # train and test loss curves
    plt.plot(graph_dict['epochs'], graph_dict['train_loss'], label='train_loss')
    plt.plot(graph_dict['epochs'], graph_dict['test_loss'], label='test_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()

    # train and test accuracies
    plt.plot(graph_dict['epochs'], graph_dict['train_acc'], label='train_acc')
    plt.plot(graph_dict['epochs'], graph_dict['test_acc'], label='test_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
    plt.close()

    if save_dict:
        j = json.dumps(graph_dict)
        f = open(os.path.join(plots_dir, "stats.json"), "w")
        f.write(j)
        f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

model_dir = os.path.join('models', 'final_cls_pcnn500_ownsub')

opt = parser.parse_args()
print (opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device: {}'.format(device))

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'data/roofn3d_data_multitask_v1',
                      task='classification',
                      npoints = opt.num_points,
                      load_in_memory=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'data/roofn3d_data_multitask_v1',
                           task='classification',
                           train = False,
                           npoints = opt.num_points,
                           load_in_memory=True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print('train: {} test: {}'.format(len(dataset), len(test_dataset)))
num_classes = len(dataset.classes)
print('classes', num_classes)


classifier = PointCNN(num_classes = num_classes).to(device)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001)

num_batch = len(dataset)/opt.batchSize
num_test_batch = len(test_dataset)/opt.batchSize
n_log = 100

epochs = []
train_acc = []
train_loss = []
test_acc = []
test_loss = []


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

    total_train_correct = 0
    total_train_loss = 0
    classifier.train()
    for i, data in enumerate(dataloader, 0):
        train_batch_num += 1
        points, target = data
        points, target = points.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = classifier((points, points))
        loss = F.cross_entropy(pred, target.long())
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_train_correct += correct.item()
        total_train_loss += loss.item()

        if i%n_log == 0:
            print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(),correct.item() / float(opt.batchSize)))

        del loss

    epoch_train_acc = float(total_train_correct) / float(len(dataset))
    epoch_train_loss = float(total_train_loss) / float(num_batch)
    print('train_accuracy: {}'.format(epoch_train_acc))
    print('train_loss: {}'.format(epoch_train_loss))

    total_test_correct = 0
    total_test_loss = 0
    classifier.eval()
    for j, data in enumerate(testdataloader, 0):
        test_batch_num += 1
        points, target = data
        points, target = points.to(device, non_blocking=True), target[:, 0].to(device, non_blocking=True)
        pred = classifier((points, points))
        loss = F.cross_entropy(pred, target.long())
        epoch_test_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        total_test_correct += pred_choice.eq(target.long().data).cpu().sum().item()
        total_test_loss += loss.item()
    epoch_test_acc = float(total_test_correct) / float(len(test_dataset))
    epoch_test_loss = float(total_test_loss) / float(num_test_batch)
    print('test accuracy: {}'.format(epoch_test_acc))
    print('test_loss: {}'.format(epoch_test_loss))

    epochs.append(epoch)
    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)
    plot_graphs({'epochs':epochs, 'train_acc':train_acc, 'train_loss':train_loss, 'test_acc':test_acc, 'test_loss':test_loss})
    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (model_dir, epoch))
