import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def plotData(data):
    # 区分数据中0 1的部分并分别绘制 具体实现分negative positive
    negative = data[data["Admitted"].isin([0])]
    positive = data[data["Admitted"].isin([1])]
    # print(negative)
    plot_x = np.linspace(data['Exam1'].min(), data['Exam1'].max(), 100)
    plot_f = (-result[0][0] - result[0][1] * plot_x) / result[0][2]  # 将式子整理为x1与x2的关系即可
    plt.figure()
    plt.plot(plot_x, plot_f, 'r', label='Prediction')
    plt.xlabel("Exam1 score")
    plt.ylabel("Exam2 score")
    plt.scatter(positive["Exam1"], positive["Exam2"], s=20, c='b', marker='x', label='Admitted')
    plt.scatter(negative["Exam1"], negative["Exam2"], s=20, c='y', marker='o', label='Admitted')
    plt.legend()  # 加上右上角的小label
    plt.show()


# function g
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 返回代价和梯度
def costFunction(X, y, theta):
    m = len(X)
    grade = np.zeros(theta.shape[0])
    hx = sigmoid(np.dot(X, theta))
    # print(theta.shape)
    cost = (1 / m) * np.sum(np.multiply(-y, np.log(hx)) - np.multiply((1 - y), np.log(1 - hx)))
    for j in range(theta.shape[0]):
        grade[j] = (1 / m) * np.sum(np.multiply((hx - y), X[:, j:j + 1]))
    return cost, grade


# 使用库函数需要定义目标函数与梯度函数
def cost(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    hx = sigmoid(X * theta.T)
    return (1 / len(X)) * np.sum(np.multiply(-y, np.log(hx)) - np.multiply((1 - y), np.log(1 - hx)))


def gradient(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)  # 啊 将其矩阵化真的很重要
    y = np.mat(y)
    para = theta.ravel().shape[1]
    grade = np.zeros(para)
    hx = sigmoid(X * theta.T)
    for j in range(para):
        grade[j] = (1 / len(X)) * np.sum(np.multiply((hx - y), X[:, j]))
    return grade


def hypothesisFunc(theta, X):
    return sigmoid(np.dot(theta.T, X))


if __name__ == '__main__':
    # plotData("ex2data1.txt")
    path = 'ex2data1.txt'
    data = pd.read_csv(path, header=None, names=["Exam1", "Exam2", "Admitted"])
    # print(data.head()) # 这个函数可以打印前5个数据
    # plotData(data)
    dt = data.copy()
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, 0: cols - 1]  # X是数据中的前两列
    y = data.iloc[:, cols - 1: cols]
    theta = np.zeros(3)  # 为啥不写明维度就会成为数组而非矩阵，进而得到计算错误的结论
    # 类型转换
    X = X.values
    y = y.values
    print(X.shape, y.shape, theta.shape)
    # print(X.head())
    # print(y.head())
    # cost, grad = costFunction(X, y, theta)
    # print(cost(theta, X, y))
    # print(gradient(theta, X, y))
    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
    # print(result[0])
    # print(cost(result[0], X, y))
    print(hypothesisFunc(result[0], [1, 45, 85]))

    plotData(dt)
