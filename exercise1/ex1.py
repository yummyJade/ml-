import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
# A = np.eye(5)
# print(A)


path = 'ex1data1.txt'
df = pd.read_csv(path, header=None, names=['population', 'profit'])

# data = np.array(df.loc[:,:])
# X = data[:, 0]
# Y = data[:, 1]
# plt.figure()
# plt.xlabel('Population of City in 10,000s')
# plt.ylabel('Profit in $10,000s')
# plt.plot(X, Y, 'x')
# plt.show()



# 其中零向量是为了theta0设计的
data = df
# 加入一列
data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, :-1]  # X是data里的除最后列
y = data.iloc[:, cols - 1:cols]  # y是data最后一列

m = data.shape[0]
# 转化为numpy矩阵
X = X.values
y = y.values
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01
#
print(X.shape, y.shape, theta.shape)


#
def computeCost(X, y, theta):
    m = len(X)
    return np.sum(np.power((np.dot(X, theta) - y), 2)) / (2 * m)


# print(computeCost(X, y, m))

para = theta.shape[0]


# 2.2 Gradient Descent

# 应该改成某一个的shape
def gradientDescent(X, y, theta, alpha, iterations):
    his = np.zeros((iterations, 1))
    temptheta = np.zeros((para, 1))
    # 先做出loop的结构

    for i in range(iterations):
        part1 = np.dot(X, theta) - y

        for j in range(para):
            temp = (alpha / len(X)) * np.sum(np.multiply(part1, X[:, j:j+1]))  # 这里的切片到底怎么写才能让维度为(97,1),神奇的是若写成X[:,j]就会使得维度为(97,) 进而无法完成运算
            # print(np.sum(np.multiply(part1, X[:,j:j+1])))
            temptheta[j,:] = theta[j,:] - temp
            # print(temptheta[j,0])

        # 再同时更新所有的theta
        theta = temptheta
        # 计算并打印代价函数
        his[i, 0] = computeCost(X, y, theta)  # woc没传值真滴要命,注意theta每次都要更新，要不断传副本
    return theta, his


theta, cost = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

print(theta.shape)
predict1 = np.dot([1,3.5], theta)
print("predict1:",predict1)
predict2 = np.dot([1,7], theta)
print("predict2:",predict2)


# 绘制拟合的直线

# 这里的范围应该是数据中的最大最小值
x = np.linspace(data['population'].min(), data['population'].max(), 100)
f = theta[0, :] + (theta[1, :] * x)
plt.figure()
plt.plot(x, f, 'r', label='Prediction')
plt.plot(data['population'], data['profit'], 'x', label='Training Data')
plt.legend(loc='best')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Predict')
plt.show()

# 2.4 Vistalizing J(theta)

