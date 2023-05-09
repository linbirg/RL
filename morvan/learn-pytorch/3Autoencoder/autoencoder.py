'''use torch to impl autoencoder
   优化了autoencoder，生成数字相对比较准确并且还原度较高，主要参考了hilton的思路，向扩展784到1024，再递减，decoder相反，最后有1024缩减到784
   下一步可以研究DALL-E https://zhuanlan.zhihu.com/p/625975291
'''

import os

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(
    ),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,  # download it if you don't have it
)

# plot one example
# print(train_data.train_data.size())  # (60000, 28, 28)
# print(train_data.train_labels.size())  # (60000)
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

BATCH_SIZE = 64
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=BATCH_SIZE,
                               shuffle=True)


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()



        # self.encoder = nn.Sequential(nn.Linear(28 * 28, 1024), nn.LeakyReLU(),nn.LayerNorm(1024)
        #                              nn.Linear(1024, 512), nn.LeakyReLU(),
        #                              nn.LayerNorm(512), nn.Linear(512, 256),
        #                              nn.LeakyReLU(), nn.LayerNorm(256),
        #                              nn.Linear(256, 128), nn.LeakyReLU(),
        #                              nn.LayerNorm(128), nn.Linear(128, 64),
        #                              nn.LeakyReLU(), nn.LayerNorm(64),
        #                              nn.Linear(64, 12))  # nn.LeakyReLU(),
        #  nn.Linear(12, 3))
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 1024), nn.LeakyReLU(),
                                     nn.LayerNorm(1024), nn.Linear(1024, 512),
                                     nn.LeakyReLU(), nn.LayerNorm(512),
                                     nn.Linear(512, 256), nn.LeakyReLU(),
                                     nn.LayerNorm(256), nn.Linear(256, 12))

        self.decoder = nn.Sequential(nn.Linear(12, 256), nn.LeakyReLU(),
                                     nn.LayerNorm(256), nn.Linear(256, 512),
                                     nn.LeakyReLU(), nn.LayerNorm(512),
                                     nn.Linear(512, 1024), nn.LeakyReLU(),
                                     nn.LayerNorm(1024),
                                     nn.Linear(1024, 28 * 28), nn.Sigmoid())

        # self.decoder = nn.Sequential(nn.Linear(12, 64), nn.LeakyReLU(),
        #                              nn.LayerNorm(64), nn.Linear(64, 128),
        #                              nn.LeakyReLU(), nn.LayerNorm(128),
        #                              nn.Linear(128, 256), nn.LeakyReLU(),
        #                              nn.LayerNorm(256), nn.Linear(256, 512),
        #                              nn.LeakyReLU(), nn.LayerNorm(512),
        #                              nn.Linear(512, 28 * 28), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)  # [B, 128] -> [B, 64] -> [B, 12]->[B, 3]
        decoded = self.decoder(encoded)
        return encoded, decoded


class Trainer():

    EPOCH = 10
    LR = 0.005  # learning rate
    PATH = './auto.pt'

    # N_TEST_IMG = 5

    def __init__(self, autoencoder=None):
        self.autoencoder = autoencoder
        if autoencoder is None or not isinstance(autoencoder, AutoEncoder):
            self.autoencoder = AutoEncoder()

        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(),
                                          lr=self.LR)
        # self.loss_func = nn.MSELoss()
        self.loss_func = nn.BCELoss()
        self.a = None

    def train(self, trainloader, show=True, iscontinue=True):

        if iscontinue and os.path.exists(trainer.PATH):
            self.load()

        for epoch in range(self.EPOCH):
            for step, (x, b_label) in enumerate(trainloader):
                b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
                b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

                _, decoded = self.autoencoder(b_x)

                loss = self.loss_func(decoded, b_y)  # mean square error
                self.optimizer.zero_grad(
                )  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if step % 100 == 0:
                    print('Epoch: ', epoch, '| Step: ', step,
                          '| train loss: %.4f' % loss.data.numpy())

                    if show:
                        self.show()

        torch.save(self.autoencoder.state_dict(), self.PATH)

    def load(self, filename=None):
        if filename is None:
            filename = self.PATH  # if filename is None, use default path of autoean.pt

        self.autoencoder.load_state_dict(torch.load(
            filename))  # load weights from file. Note that the file is binary-
        self.autoencoder = self.autoencoder.eval()

    def show(self, start=5, reset=False):
        # initialize figure
        N_IMG = 5

        if reset or self.a is None:
            f, a = plt.subplots(2, N_IMG, figsize=(5, 2))
            self.a = a
            plt.ion()  # continuously plot

        a = self.a

        # original data (first row) for viewing
        view_data = train_data.train_data[start:start + N_IMG].view(
            -1, 28 * 28).type(torch.FloatTensor) / 255.

        for i in range(N_IMG):
            a[0][i].clear()
            a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)),
                           cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        # plotting decoded image (second row)
        _, decoded_data = self.autoencoder(view_data)
        for i in range(N_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)),
                           cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        plt.draw()
        plt.pause(0.05)
        plt.show()


if __name__ == '__main__':
    trainer = Trainer()
    # if os.path.exists(trainer.PATH):
    #     trainer.load()
    # else:
    #     trainer.train(trainloader=train_loader)
    trainer.train(trainloader=train_loader)

    import random as rd
    import time

    while True:
        length = train_data.train_data.size()[0] - 5
        pos = rd.randint(0, length)
        trainer.show(start=pos)
        time.sleep(3)

# autoencoder = AutoEncoder()

# loss_func = nn.MSELoss()

# original data (first row) for viewing
# view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(
#     torch.FloatTensor) / 255.

# for i in range(N_TEST_IMG):
#     a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)),
#                    cmap='gray')
#     a[0][i].set_xticks(())
#     a[0][i].set_yticks(())

# for epoch in range(EPOCH):
#     for step, (x, b_label) in enumerate(train_loader):
#         b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
#         b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

#         encoded, decoded = autoencoder(b_x)

#         loss = loss_func(decoded, b_y)  # mean square error
#         optimizer.zero_grad()  # clear gradients for this training step
#         loss.backward()  # backpropagation, compute gradients
#         optimizer.step()  # apply gradients

#         if step % 100 == 0:
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

#             # plotting decoded image (second row)
#             _, decoded_data = autoencoder(view_data)
#             for i in range(N_TEST_IMG):
#                 a[1][i].clear()
#                 a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i],
#                                           (28, 28)),
#                                cmap='gray')
#                 a[1][i].set_xticks(())
#                 a[1][i].set_yticks(())
#             plt.draw()
#             plt.pause(0.05)

# plt.ioff()
# plt.show()

# # visualize in 3D plot
# view_data = train_data.train_data[:200].view(-1, 28 * 28).type(
#     torch.FloatTensor) / 255.
# encoded_data, _ = autoencoder(view_data)
# fig = plt.figure(2)
# ax = Axes3D(fig)
# X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(
# ), encoded_data.data[:, 2].numpy()
# values = train_data.train_labels[:200].numpy()
# for x, y, z, s in zip(X, Y, Z, values):
#     c = cm.rainbow(int(255 * s / 9))
#     ax.text(x, y, z, s, backgroundcolor=c)
# ax.set_xlim(X.min(), X.max())
# ax.set_ylim(Y.min(), Y.max())
# ax.set_zlim(Z.min(), Z.max())
# plt.show()
