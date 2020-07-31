import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plotData(data):
    negative = data[data['Accepted'].isin([0])]
    postive = data[data['Accepted'].isin([1])]
    # print(negative.head())
    plt.figure()
    plt.scatter(negative['Test1'], negative['Test2'], s=25, c='b', marker='+', label='y=0')
    plt.scatter(postive['Test1'], postive['Test2'], s=25, c='r', marker='o', label='y=1')
    x1,y1  = find_decison_boundary(result)
    plt.scatter(x1, y1, c='y', s=10, marker='o', label='Prediction')
    plt.xlabel('Microchip Test1')
    plt.ylabel('Microchip Test2')
    plt.legend(loc='best')
    plt.show()


# 原来的题目中直接提供了，完成原来两个特征的映射
def mapFeature(x1, x2):

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            data['F' + str(i - j) + str(j)] = np.power(x1, i - j) * np.power(x2, j)

    data.drop('Test1', axis=1, inplace=True)
    data.drop('Test2', axis=1, inplace=True)


# function g
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, X, y, lam):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    hx = sigmoid(X * theta.T)
    # 注意正则项没有theta 0
    thetalen = theta.shape[1]
    reg = (lam / (2 * len(X))) * np.sum(np.power(theta[:, 1:thetalen], 2))
    return (1 / len(X)) * np.sum(np.multiply(-y, np.log(hx)) - np.multiply((1 - y), np.log(1 - hx))) + reg


def gradient(theta, X, y, lam):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    para = int(theta.ravel().shape[1])
    grade = np.zeros(para)
    hx = sigmoid(X * theta.T)
    # 单独更新j=0部分
    for j in range(para):
        reg = (lam / len(X)) * theta[:, j]
        t = (1 / len(X)) * np.sum(np.multiply((hx - y), X[:, j]))
        if j == 0:
            grade[j] = t
        else:
            grade[j] = t + reg
    return grade


def hypothesisFunc(theta, x1, x2):
    # 这个式子和最开始定义的那个映射相关？我没看出来这个是哪里来的，故照着打了一遍
    temp = theta[0][0]
    place = 0
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            temp += np.power(x1, i - j) * np.power(x2, j) * theta[0][place + 1]
            place += 1
    return temp

def find_decison_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)
    cordinates = [(x, y) for x in t1 for y in t2]   # 这干嘛的
    # zip函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象。
    x_cord, y_cord = zip(*cordinates)
    # print(x_cord)
    # dataFrame可以建表，建完以后就是我们平时用pandas读数据时的模样
    # 这一步容易理解，就是把点以及该点的预测值存在表中
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hypothesisFunc(theta, h_val['x1'], h_val['x2'])
    decision = h_val[np.abs(h_val['hval'] < 2 * 10**-3)]        # 这一步又是什么
    return decision.x1, decision.x2

if __name__ == '__main__':
    path = 'ex2data2.txt'
    degree = 6
    data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])
    # print(data.head())
    dt = data.copy()
    # plotData(data)

    data.insert(3, 'Ones', 1)
    mapFeature(data['Test1'], data['Test2'])
    # print(data.head())
    # 整理出数据
    cols = data.shape[1]
    X = data.iloc[:, 1:cols]
    y = data.iloc[:, 0:1]
    theta = np.zeros(cols - 1)
    # 转换
    X = X.values
    y = y.values
    print(X.shape, y.shape, theta.shape)  # 搞清楚矩阵的维度关系真的非常重要
    print(np.mat(theta).shape)
    lam = 1
    # c = cost(theta, X, y, lam)
    # print(c)
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y, lam))
    print(result)
    plotData(dt)
