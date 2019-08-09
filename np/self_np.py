import numpy as np

# sigmoid函数
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# 输入数据集
X = np.array([  [0,1],
                [1,1],
                [0,1],
                [1,1],
				[1,0] ])

# 输出数据集
y = np.array([[0,0,1,1,0]]).T

# 设置随机数种子使计算结果是确定的
# （实践中这是一个很好的做法）
np.random.seed(1)

# 随机初始化权重（均值0）
syn0 = 2*np.random.random((2,1)) - 1
print("syn0:",syn0)
for iter in range(10000):

    # 前向传播
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # 差多少？
    l1_error = y - l1

    # 误差乘以
    # sigmoid在l1处的斜率
    l1_delta = l1_error * nonlin(l1,True)

    # 更新权重
    syn0 += np.dot(l0.T,l1_delta)

print("训练后输出：")
print(l1)