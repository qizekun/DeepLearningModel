from timm import models
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import os
import json
import time
from utils.transforms import get_transforms
from module.backbone import ResNet50
from utils.tools import AccuracyMeter, TenCropsTest, data_analysis, Net
from torchtoolbox.nn import LabelSmoothingLoss


class Classification:
    """
        Parameters:
        --------
            model : str
        Returns:
        --------
            module : nn.module
        Example:
        --------
            model = Classification(model='resnet')
            model.lr = 1e-3
            model.batch_size = 16
            model.epochs = 500
            model.load_dataset()
            model.describe = 'test'
            model.train()
    """

    def __init__(self, model):
        self.model_name = model
        self.net = None

        # parameter
        self.batch_size = 32
        self.lr = 1e-3
        self.epochs = 200
        self.describe = ''
        self.gpu = '0'
        self.seed = 2022

        # data info
        self.dataset_path = 'flower'
        self.path = ''
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 0

        # train info
        self.train_info = {
            'all_val_accurate': [],
            'all_train_loss': [],
            'all_val_loss': [],
            'epoch': 0,
            'best_acc': 0.0,
            'best_epoch': 0
        }

        # 损失函数 默认用交叉熵
        self.LabelSmoothing = 0.0
        self.loss_function = nn.CrossEntropyLoss()
        # 优化器
        self.optimizer = 'sgd'
        # device: GPU or CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # load pretrain
        self.pretrain = False
        self.pretrain_path = ''
        # save model
        self.save = False

    def init_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    def load_dataset(self, dataset_path='flower'):
        self.dataset_path = dataset_path
        self.path = f'{self.dataset_path}/{self.model_name}'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if not os.path.exists(self.path + '/csv/'):
            os.mkdir(self.path + '/csv/')
        if not os.path.exists(self.path + '/pic/'):
            os.mkdir(self.path + '/pic/')

        # 数据增强
        data_transforms = get_transforms(resize_size=256, crop_size=224)
        self.train_dataset = datasets.ImageFolder(root=self.dataset_path + "/data/train",
                                                  transform=data_transforms["train"])
        self.val_dataset = datasets.ImageFolder(root=self.dataset_path + "/data/val",
                                                transform=data_transforms["val"])
        if os.path.exists(self.dataset_path + "/data/test"):
            test_path = self.dataset_path + "/data/test"
        else:
            test_path = self.dataset_path + "/data/val"
        self.test_dataset = {
            'test' + str(i):
                datasets.ImageFolder(
                    root=test_path,
                    transform=data_transforms["test" + str(i)]
                )
            for i in range(10)
        }

        class_list = self.train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in class_list.items())
        json_str = json.dumps(cla_dict, indent=4)
        with open(self.path + '/class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        self.num_classes = len(cla_dict.keys())

    def get_net(self):
        if self.pretrain:
            # pretrain_path为空，加载timm预训练模型
            if self.pretrain_path == '':
                net = self.pretrain_model(self.num_classes, True)
                net = Net(net, self.num_classes, self.model_name)
                self.describe += f'pretrain_official'
            # pretrain_path不为空，加载本地预训练模型
            else:
                net = self.pretrain_model(self.num_classes, False)
                net.load_state_dict(torch.load(self.pretrain_path), False)
                self.describe += f'pretrain_self'
        else:
            net = self.pretrain_model(self.num_classes, False)
            self.describe += 'scratch'
        return net

    def pretrain_model(self, num_classes, pretrained):
        if self.model_name == 'resnet':
            net = ResNet50(pretrained=pretrained, num_classes=num_classes)
        elif self.model_name == 'convnext':
            net = models.convnext.convnext_base_in22ft1k(pretrained=pretrained, num_classes=num_classes)
        elif self.model_name == 'vit':
            net = models.vit_base_patch16_224_in21k(pretrained=pretrained, num_classes=num_classes)
        elif self.model_name == 'mlp':
            net = models.mixer_b16_224_in21k(pretrained=pretrained, num_classes=num_classes)
        elif self.model_name == 'deit':
            net = models.deit_base_distilled_patch16_224(pretrained=pretrained, num_classes=num_classes)
        elif self.model_name == 'swin':
            net = models.swin_base_patch4_window7_224_in22k(pretrained=pretrained, num_classes=num_classes)
        else:
            raise Exception('没有该模型')
        return net

    def train(self):

        # 初始化随机种子
        self.init_seed()
        if self.train_dataset is None:
            self.load_dataset(self.dataset_path)
        self.net = self.get_net()
        self.net.to(self.device)

        # dataloader
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=False)
        test_loaders = {
            'test' + str(i):
                torch.utils.data.DataLoader(
                    self.test_dataset["test" + str(i)], batch_size=4, shuffle=False, num_workers=4
                )
            for i in range(10)
        }

        # optimizer
        if self.pretrain:
            params = [
                {"params": filter(lambda p: p.requires_grad, self.net.backbone.parameters()), "lr": self.lr},
                {"params": filter(lambda p: p.requires_grad, self.net.head.parameters()), "lr": self.lr * 10}
            ]
        else:
            params = self.net.parameters()
        if self.optimizer == 'radam':
            optimizer = optim.RAdam(params, lr=self.lr, weight_decay=5e-4)
        elif self.optimizer == 'adamw':
            optimizer = optim.AdamW(params, lr=self.lr, weight_decay=5e-4)
        else:
            optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=4 if self.pretrain else 20,
                                                               verbose=True, threshold=1e-3, min_lr=1e-6,
                                                               cooldown=4 if self.pretrain else 20)

        # loss
        if self.LabelSmoothing > 0:
            self.loss_function = LabelSmoothingLoss(classes=self.num_classes, smoothing=self.LabelSmoothing)

        save_path = self.path + f'{self.model_name}.pth'

        # 开始进行训练和测试
        train_len = len(train_loader) - 1
        iter_nums = train_len * self.epochs
        running_loss = step = epoch = 0
        t = time.perf_counter()
        for iter_num in range(iter_nums):
            self.net.train()

            if iter_num % train_len == 0:
                train_iter = iter(train_loader)

            train_inputs, train_labels = next(train_iter)
            train_inputs, train_labels = train_inputs.to(self.device), train_labels.to(self.device)
            train_outputs = self.net(train_inputs)
            if isinstance(train_outputs, tuple):
                train_outputs = train_outputs[0]
            loss = self.loss_function(train_outputs, train_labels)

            self.net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            step += 1
            rate = step / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtraining: {:^3.0f}%[{}->{}]".format(int(rate * 100 + 0.5), a, b), end="")

            if iter_num % train_len == 0 and iter_num != 0:
                epoch += 1
                print('\n', time.perf_counter() - t)
                acc_meter = AccuracyMeter(topk=(1,))
                with torch.no_grad():
                    self.net.eval()
                    val_loss = 0.0
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels = val_inputs.to(self.device), val_labels.to(self.device)
                        val_outputs = self.net(val_inputs)
                        val_loss += self.loss_function(val_outputs, val_labels)
                        acc_meter.update(val_outputs, val_labels)
                    val_accurate = acc_meter.avg[1]
                    acc_meter.reset()
                    if val_accurate > self.train_info['best_acc']:
                        self.train_info['best_acc'] = val_accurate
                        self.train_info['best_epoch'] = epoch
                        if self.save:
                            torch.save(self.net.state_dict(), save_path)

                    scheduler.step(val_accurate)  # 动态调整学习率
                    self.train_info['all_val_accurate'].append(val_accurate / 100)
                    self.train_info['all_train_loss'].append(running_loss / len(train_loader))
                    self.train_info['all_val_loss'].append(val_loss / len(val_loader))
                    print('[epoch %d] train_loss: %.3f val_loss: %.3f  test_accuracy: %.3f' %
                          (epoch, running_loss / len(train_loader), val_loss / len(val_loader), val_accurate))
                    self.train_info['epoch'] = epoch
                    data_analysis(self)
                    running_loss = step = 0
                    t = time.perf_counter()

        test_acc = TenCropsTest(test_loaders, self.net)
        print(f'Finished Training! Test acc: {test_acc}')


if __name__ == '__main__':
    model = Classification(model='resnet')
    model.lr = 1e-3
    model.batch_size = 32
    model.epochs = 200
    model.pretrain = False
    model.gpu = "0"
    model.optimizer = 'radam'
    # model.LabelSmoothing = 0.1

    model.load_dataset('flower')
    model.train()
