import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel:1;输出channel:6;5*5卷积核
        self.conv1 = nn.Conv2d(
            1, 6, 5)  # input channels, output channels, kernel size.
        self.conv2 = nn.Conv2d(
            6, 16, 5)  # input channels, output channels, kernel size.
        # 仿射变换：y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # input size, output size. (5*5)
        self.fc2 = nn.Linear(
            120, 84)  # input size, output size. (84  # 输出维度为1 （一个图像只有
        self.fc3 = nn.Linear(84, 10)  # input size, output size. (10  # 输出维度

    def forward(self, x):
        # 2X2 max pooling.
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(
            x))  # fully connected. (84)  # output size. (84)  # output
        x = F.relu(self.fc2(x))  # fully connected
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):  # 计算维度，不一定是84(84 * 5*5)
        size = x.size()[
            1:]  # all dimensions except the batch dimension. (16,5,5)->(16,25)
        num_features = 1  # 维度是1. (1,25)->(1,1) (1,1)->(
        for s in size:  # 计算维度的次数. (5*5)->(1,25) (25)->
            num_features *= s  # (1,25)->(1,1) (1,1)->(1) (1)
        return num_features  # (1) (1,25)->(1,84) (84)->(1,10) (10


# 优化器
import torch.optim as optim

net = Net()

# 尝试一个随机32*32的输入
input = torch.randn(1, 1, 32, 32)

optimizer = optim.SGD(net.parameters(), lr=0.1)  # 学习率0.1
optimizer.zero_grad()  # 清楚梯度所有的变化. (1,84)->(1,10) (1,

out = net(input)
target = torch.randn(10)
target = target.view(1, -1)  #
criterion = nn.MSELoss()

loss = criterion(out, target)  # 计算Loss.
loss.backward()
optimizer.step()  # 更新参数.

print(loss)