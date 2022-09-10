import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import json
import os
from module.PCT import Pct
from data.ModelNet40 import ModelNet40
from utils.tools import init_seed, AccuracyMeter, TenCropsTest, data_analysis, get_params
from main import train_one_iter, val_one_iter, train_model


class Cls3d:
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
        self.dataset = 'modelnet40'
        self.data_path = 'modelnet40'
        self.work_dir = 'work_dir/cls3d'
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 1000
        self.num_points = 1024

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
        self.loss_function = 'CrossEntropy'
        # 优化器
        self.optimizer = 'sgd'
        # device: GPU or CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # load pretrain
        self.pretrain = False
        self.pretrain_path = ''
        # save model
        self.save = False

    def load_dataset(self, data_path):
        self.work_dir = f'{self.work_dir}/{self.dataset}/{self.model_name}'
        os.makedirs(self.work_dir + '/csv/', exist_ok=True)
        os.makedirs(self.work_dir + '/pic/', exist_ok=True)

        if self.dataset == 'modelnet40':
            self.train_dataset = ModelNet40(partition='train', num_points=self.num_points, data_path=data_path)
            self.val_dataset = ModelNet40(partition='test', num_points=self.num_points, data_path=data_path)
        # class_list = self.train_dataset.class_to_idx
        # cla_dict = dict((val, key) for key, val in class_list.items())
        # json_str = json.dumps(cla_dict, indent=4)
        # with open(self.work_dir + '/class_indices.json', 'w') as json_file:
        #     json_file.write(json_str)
        # self.num_classes = len(cla_dict.keys())

    def create_model(self):
        if self.model_name == 'pct':
            net = Pct()
        else:
            raise f'No model {self.model_name}'
        return net.to(self.device)

    def train(self):
        init_seed(self.seed)
        if self.train_dataset is None or self.val_dataset is None:
            self.load_dataset(self.data_path)
        train_loader = DataLoader(self.train_dataset, num_workers=4, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_dataset, num_workers=4, batch_size=self.batch_size,
                                shuffle=True, drop_last=False)

        self.net = self.create_model()
        self.net = nn.DataParallel(self.net)

        params = get_params(self.pretrain, self.net, self.lr)
        if self.optimizer == 'radam':
            optimizer = torch.optim.RAdam(params, lr=self.lr, weight_decay=5e-4)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1,
                                                               patience=4 if self.pretrain else 20,
                                                               verbose=True, threshold=1e-3, min_lr=1e-6,
                                                               cooldown=4 if self.pretrain else 20)

        loss_function = nn.CrossEntropyLoss(label_smoothing=self.LabelSmoothing)
        save_path = f'{self.work_dir}/{self.model_name}.pth'
        # 开始进行训练和测试
        train_model(self, train_loader, val_loader, None, self.epochs, self.net, self.device, loss_function,
                    optimizer, scheduler, self.train_info, self.save, save_path, test=False)


if __name__ == '__main__':
    model = Cls3d(model='pct')
    model.lr = 1e-2
    model.batch_size = 64
    model.epochs = 200
    model.dataset = 'modelnet40'
    model.load_dataset(data_path='../../data/modelnet40')
    model.train()
