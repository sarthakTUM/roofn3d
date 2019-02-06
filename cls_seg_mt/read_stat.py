import json
import os


def max_val_accuracy(stats):
    # find max val accuracy, corresponding train accuracy, epoch and losses
    val_acc = stats['val_acc']
    max_val_acc_idx = val_acc.index(max(val_acc))
    max_val_acc = max(val_acc)
    train_acc = stats['train_acc'][max_val_acc_idx]
    val_loss = stats['val_loss'][max_val_acc_idx]
    train_loss = stats['train_loss'][max_val_acc_idx]
    return {'epoch':max_val_acc_idx,
            'max_val_acc':max_val_acc,
            'train_acc':train_acc,
            'val_loss':val_loss,
            'train_loss':train_loss}

def min_val_loss(stats):
    # find max val accuracy, corresponding train accuracy, epoch and losses
    val_loss = stats['val_loss']
    min_val_loss_idx = val_loss.index(min(val_loss))
    min_val_loss = min(val_loss)
    train_loss = stats['train_loss'][min_val_loss_idx]
    val_cls_acc = stats['val_cls_acc'][min_val_loss_idx]
    val_cls_loss = stats['val_cls_loss'][min_val_loss_idx]
    val_seg_acc = stats['val_seg_acc'][min_val_loss_idx]
    val_seg_loss = stats['val_seg_loss'][min_val_loss_idx]
    return {'epoch':min_val_loss_idx,
            'min_val_loss':min_val_loss,
            'train_loss':train_loss,
            'val_cls_acc':val_cls_acc,
            'val_cls_loss':val_cls_loss,
            'val_seg_acc':val_seg_acc,
            'val_seg_loss':val_seg_loss}

def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    stats = read_json(os.path.join('models', 'final_damage', 'seg', 'plots', 'stats.json'))
    # epoch, val_acc, train_acc, val_loss, train_loss = max_val_accuracy(stats)
    # print('epoch: {} val_acc: {} val_loss: {} train_acc: {} train_loss: {}'.format(epoch, val_acc, val_loss, train_acc, train_loss))
    d = max_val_accuracy(stats)
    print(d)