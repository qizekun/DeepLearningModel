import torch
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import numpy as np
import random


__all__ = ['init_seed', 'AccuracyMeter', 'TenCropsTest', 'data_analysis', 'query_params', 'get_params']


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def query_params(query, parameters):
    for p in list(parameters):
        if p is query:
            return True
    return False


def get_params(pretrain, net, lr):
    if pretrain:
        if hasattr(net, 'head'):
            params = [
                {"params": filter(lambda p: p.requires_grad and not query_params(p, net.head.parameters()),
                                  net.parameters()), "lr": lr},
                {"params": filter(lambda p: p.requires_grad, net.head.parameters()), "lr": lr * 10}
            ]
        elif hasattr(net, 'fc'):
            params = [
                {"params": filter(lambda p: p.requires_grad and not query_params(p, net.fc.parameters()),
                                  net.parameters()), "lr": lr},
                {"params": filter(lambda p: p.requires_grad, net.fc.parameters()), "lr": lr * 10}
            ]
        else:
            print('Use same lr')
            params = net.parameters()
    else:
        params = net.parameters()
    return params

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class OnlineMeter(object):
    """Computes and stores the average and variance/std values of tensor"""

    def __init__(self):
        self.mean = torch.FloatTensor(1).fill_(-1)
        self.M2 = torch.FloatTensor(1).zero_()
        self.count = 0.
        self.needs_init = True

    def reset(self, x):
        self.mean = x.new(x.size()).zero_()
        self.M2 = x.new(x.size()).zero_()
        self.count = 0.
        self.needs_init = False

    def update(self, x):
        self.val = x
        if self.needs_init:
            self.reset(x)
        self.count += 1
        delta = x - self.mean
        self.mean.add_(delta / self.count)
        delta2 = x - self.mean
        self.M2.add_(delta * delta2)

    @property
    def var(self):
        if self.count < 2:
            return self.M2.clone().zero_()
        return self.M2 / (self.count - 1)

    @property
    def std(self):
        return self.var().sqrt()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AccuracyMeter(object):
    """Computes and stores the average and current topk accuracy"""

    def __init__(self, topk=(1,)):
        self.topk = topk
        self.reset()

    def reset(self):
        self._meters = {}
        for k in self.topk:
            self._meters[k] = AverageMeter()

    def update(self, output, target):
        n = target.nelement()
        acc_vals = accuracy(output, target, self.topk)
        for i, k in enumerate(self.topk):
            self._meters[k].update(acc_vals[i])

    @property
    def val(self):
        return {n: meter.val for (n, meter) in self._meters.items()}

    @property
    def avg(self):
        return {n: meter.avg for (n, meter) in self._meters.items()}

    @property
    def avg_error(self):
        return {n: 100. - meter.avg for (n, meter) in self._meters.items()}


def TenCropsTest(loader, net):
    with torch.no_grad():
        net.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in trange(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].cuda()
            labels = labels.cuda()
            outputs = []
            for j in range(10):
                output = net(inputs[j])
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        acc_meter = AccuracyMeter(topk=(1,))
        acc_meter.update(all_outputs, all_labels)

    return acc_meter.avg[1]


def data_analysis(model):
    all_val_accurate, all_train_loss, all_val_loss, epoch, best_acc, best_epoch = model.train_info.values()

    plt.figure(figsize=(10, 6), dpi=244)
    plt.ylim((0, 1.2))
    l1, = plt.plot(range(0, len(all_val_accurate)), all_val_accurate)
    l2, = plt.plot(range(0, len(all_train_loss)), all_train_loss)
    l3, = plt.plot(range(0, len(all_val_loss)), all_val_loss)

    columns = ['val_accurate', 'train_loss', 'val_loss']
    plt.legend(handles=[l1, l2, l3], labels=columns, loc='best')  # best表示自动分配最佳位置
    data = {
        'val_accurate': all_val_accurate,
        'train_loss': all_train_loss,
        'val_loss': all_val_loss
    }
    plt.title(f'{model.model_name}_lr={model.lr}_bs={model.batch_size}\n'
              f'best_acc={"%.4f" % best_acc}   best_epoch={best_epoch}')

    if epoch % 10 == 0 and epoch > 0:
        pd.DataFrame(data, columns=columns).to_csv(
            model.work_dir + '/csv/' + f'lr={model.lr}_batch_size={model.batch_size}_{model.describe}.csv', index=False)
    plt.savefig(model.work_dir + '/pic/' + f'lr={model.lr}_batch_size={model.batch_size}_{model.describe}.png')
    plt.close()
