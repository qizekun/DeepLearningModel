import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import json
from PIL import Image
from utils.transforms import get_transforms
from module.backbone import resnet50
from utils.tools import init_seed, TenCropsTest, get_params
from main import train_model


class Cls2d:
    """
        Parameters:
        --------
            model : str
        Returns:
        --------
            module : nn.module
        Example:
        --------
            model = Cls2d()
            model.lr = 1e-3
            model.batch_size = 16
            model.epochs = 500
            model.load_dataset()
            model.describe = 'test'
            model.train()
    """

    def __init__(self):

        self.model_name = "resnet50"
        self.net = None

        # parameter
        self.batch_size = 32
        self.lr = 1e-3
        self.epochs = 200
        self.describe = ''
        self.gpu = '0'
        self.seed = 2022

        # data info
        self.dataset = 'flowers'
        self.data_path = 'flowers'
        self.work_dir = 'work_dir/cls2d'
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = 1000
        self.resize_size = 256
        self.crop_size = 224

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
        self.scheduler = 'step'
        # device: GPU or CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        # load ckpt
        self.ckpt_path = ''
        # save model
        self.save = False


    def load_dataset(self, data_path='flowers'):
        self.data_path = data_path
        self.work_dir = f'{self.work_dir}/{self.dataset}/{self.model_name}'
        os.makedirs(self.work_dir + '/csv/', exist_ok=True)
        os.makedirs(self.work_dir + '/pic/', exist_ok=True)

        # 数据增强
        data_transforms = get_transforms(resize_size=self.resize_size, crop_size=self.crop_size)
        self.train_dataset = datasets.ImageFolder(root=self.data_path + "/train",
                                                  transform=data_transforms["train"])
        self.val_dataset = datasets.ImageFolder(root=self.data_path + "/val",
                                                transform=data_transforms["val"])
        if os.path.exists(self.data_path + "/test"):
            test_path = self.data_path + "/test"
        else:
            test_path = self.data_path + "/val"
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
        with open(self.work_dir + '/class_indices.json', 'w') as json_file:
            json_file.write(json_str)
        self.num_classes = len(cla_dict.keys())

    def create_model(self):
        net = resnet50(num_classes=self.num_classes)
        if self.ckpt_path != '':
            net.load_state_dict(torch.load(self.ckpt_path))
        self.net = net.to(self.device)

    def train(self):
        # 初始化随机种子
        init_seed(self.seed)

        # dataloader
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=False)
        val_loader = DataLoader(self.val_dataset, batch_size=8,
                                shuffle=False, num_workers=4, pin_memory=False)
        test_loaders = {
            'test' + str(i):
                DataLoader(
                    self.test_dataset["test" + str(i)], batch_size=4, shuffle=False, num_workers=4
                )
            for i in range(10)
        }

        # optimizer
        params = get_params(pretrain=False, net=self.net, lr=self.lr)
        if self.optimizer == 'radam':
            optimizer = torch.optim.RAdam(params, lr=self.lr, weight_decay=5e-4)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

        # scheduler
        if self.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10,
                                                                   verbose=True, threshold=1e-3, min_lr=1e-6, cooldown=10)

        # loss
        loss_function = nn.CrossEntropyLoss(label_smoothing=self.LabelSmoothing)

        save_path = f'{self.work_dir}/{self.model_name}.pth'
        # 开始进行训练和测试
        train_model(self, train_loader, val_loader, test_loaders, self.epochs, self.net, self.device, loss_function,
                    optimizer, scheduler, self.train_info, self.save, save_path)

    def test(self):
        # 初始化随机种子
        init_seed(self.seed)
        self.net.eval()
        if self.test_dataset is None:
            self.load_dataset(self.data_path)
        test_loaders = {
            'test' + str(i):
                torch.utils.data.DataLoader(
                    self.test_dataset["test" + str(i)], batch_size=4, shuffle=False, num_workers=4
                )
            for i in range(10)
        }

        test_acc = TenCropsTest(test_loaders, self.net)
        print(f'Finished Test! Test acc: {test_acc}')

    def inference(self, image_path):
        # 初始化随机种子
        init_seed(self.seed)
        self.net.eval()
        image = Image.open(image_path).convert('RGB')
        inference_transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = inference_transform(image).unsqueeze(0).to(self.device)
        feature = self.net.forward_features(tensor)
        result = self.net(tensor)
        return feature, result


if __name__ == '__main__':

    model = Cls2d()
    model.lr = 1e-4
    model.batch_size = 32
    model.epochs = 30
    model.pretrain = True
    model.gpu = "0"
    model.optimizer = 'sgd'
    model.scheduler = 'cos'
    # model.LabelSmoothing = 0.1
    model.save = False

    model.dataset = 'cars'
    model.load_dataset('data/cars')
    model.create_model()
    model.train()

    # model = Cls2d()
    # model.ckpt_path = 'work_dir/cls2d/flowers/resmlp_12_distilled_224/resmlp_12_distilled_224.pth'
    # model.load_dataset('data/flowers')
    # model.create_model()
    # model.test()

    # model = Cls2d()
    # model.ckpt_path = 'work_dir/cls2d/flowers/resmlp_12_distilled_224/resmlp_12_distilled_224.pth'
    # model.create_model()
    # feature, result = model.inference(image_path='data/flowers/val/daisy/5673728_71b8cb57eb.jpg')
    # print(feature)
    # print(result)
