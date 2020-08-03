import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt
from scipy.io import loadmat
from sklearn.metrics import classification_report  # 这个包是评价报告


# 将一维向量映射为20x20
def displayData(sample_images):
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
    # 构建二维矩阵，将每一个一维向量映射为20x20，matshow是矩阵的绘制方法
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()


# function g
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lam):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    hx = sigmoid(X * theta.T)
    # 注意正则项没有theta 0
    thetalen = theta.shape[1]
    reg = (lam / (2 * len(X))) * np.sum(np.power(theta[:, 1:thetalen], 2))
    j = (1 / len(X)) * np.sum(np.multiply(-y, np.log(hx)) - np.multiply((1 - y), np.log(1 - hx))) + reg
    return j


def gradient(theta, X, y, lam):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    hx = sigmoid(X * theta.T)
    reg = (lam / len(X)) * theta.T
    grade = (1 / len(X)) * (X.T * (hx - y)) + reg
    # grade0的部分不做正则化
    # print((X.T * (hx-y.T)).shape)
    grade[0, :] = (1 / len(X)) * np.sum(np.multiply((hx - y), X[:, 0]))
    # print(grade.shape) # 401x5000 有问题
    return grade


# 雅克布矩阵
def one_vs_all(X, y, K, lam):
    # y最终被分为两类，属于class k即为1，不属于class k 即为0，其中classk = {0，...K}
    rows, cols = X.shape
    theta_all = np.zeros((K, cols))
    for i in range(K):
        # yi{0,1} 属于class k时为1,而k则通过循环变量i的方式找到，每一次运算的时候对y向量进行整理
        yi = np.array([1 if label == i + 1 else 0 for label in y])
        yi = np.reshape(yi, (rows, 1))
        thetai = np.zeros(cols)  # 参数x0是一维向量
        result = opt.minimize(fun=cost, x0=thetai, method='TNC', jac=gradient, args=(X, yi, lam))
        theta_all[i, :] = result.x
    return theta_all


def prediction_one_vs_all(all_theta, X, y):
    rows, cols = X.shape
    hx = sigmoid(np.dot(X, all_theta.T))
    result_y = np.zeros(rows)

    sum = 0
    # 挑选出概率最大的为预测结果
    for i in range(rows):
        result_y[i] = np.argmax(hx[i, :]) + 1
        if y[i, 0] == result_y[i]:
            sum = sum + 1
    # 计算准确率
    accuracy = sum / rows
    return result_y


def ex3_nn(theta1, theta2, X, y):
    # 这里需要回顾一下那张图
    a1 = X
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(a2.shape[0]), axis=1)
    z3 = np.dot(a2, theta2.T)
    a3 = sigmoid(z3)
    print(a3)
    predict_y2 = np.argmax(a3, axis=1) + 1
    # print(predict_y2.shape)
    print(classification_report(y, predict_y2))


if __name__ == '__main__':
    path = "ex3data1.mat"
    # 这个文件数据读出来直接就是矩阵了
    data = loadmat(path)
    # print(data)
    # 随机选择100个样本
    # arange返回一个有重点和起点的固定步长的排列
    # 这个方式就是列出所有的X排列，并随机选取100
    sampleIndex = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample = data['X'][sampleIndex, :]  # 这个采样方式挺神奇
    # print(sample.shape)
    # displayData(sample)
    K = 10
    lam = 1
    X = data['X']
    rows, cols = X.shape
    X = np.insert(X, 0, values=1, axis=1)
    y = data['y']
    # print(y)
    theta_all = np.zeros((K, X.shape[1]))
    print(X.shape, y.shape, theta_all.shape)
    all_theta = one_vs_all(X, y, 10, 1)
    # print(all_theta)
    predict_y = prediction_one_vs_all(all_theta, X, y)
    # 哇现在的包也太全了吧，果然实验的门槛降低了
    print(classification_report(y, predict_y))

    # 现在开始神经网络
    path2 = "ex3weights.mat"
    weight = loadmat(path2)
    theta1, theta2 = weight['Theta1'], weight['Theta2']
    print(theta1, theta2)
    ex3_nn(theta1, theta2, X, y)


