"""
训练一个图片分类器,我们将按顺序做以下步骤：

通过torchvision加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
定义卷积神经网络
定义损失函数
利用训练数据训练网络
利用测试数据测试网络
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 定义一个类覆盖CIFAR10中的url
class MyCIFAR10(torchvision.datasets.CIFAR10):
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def get_cifar10_data_set_and_loader(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar_set = MyCIFAR10(root='./data',
                          train=train,
                          download=True,
                          transform=transform)  #dat

    cifar_loader = torch.utils.data.DataLoader(cifar_set,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=2)

    return cifar_set, cifar_loader


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


# 输出图像的函数
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    trainset, trainloader = get_cifar10_data_set_and_loader()

    for data in trainset:
        images, labels = data  # 取出一个batch的数据和标签，可以从数据迭代器中取数据

        print(labels, type(labels))
        # ## 随机获取训练图片
        # ## 显示图片
        print(classes[labels])
        imshow(torchvision.utils.make_grid(images))
        # ## 打印图片标签
