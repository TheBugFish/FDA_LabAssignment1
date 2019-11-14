import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def convertLabel(l):
    labelString = l.decode()
    return 0 if labelString == 'Iris-setosa' else 1


def hypothesis(sample, theta):

    res = [(1 / (1 + math.exp(-i * theta))) for i in sample]
    print(res)
    return sum(res)


def loss(sampleset, labelset, theta):
    result = 0
    index = 0
    for i in sampleset:
        result += -labelset[index] * math.log(hypothesis(i, theta)) - (
            1 - labelset[index]) * math.log(1 - hypothesis(i, theta))
        index += 1
    return result


def gradient(sampleset, labelset, theta):
    result = 0
    index = 0
    for i in sampleset:
        result += ((hypothesis(i, theta) - labelset[index]) * i)


def getRandomNumbers():


def drawnSamples(samples, randomNumbers):


def drawnLabels(labels, randomNumbers):


def SGD(samples, labels):
    theta = 0
    T = 100
    t = 0
    learningRate = 100
    sampleAmount = 10
    randomSamples
    while t < T:
        randomNumbers = getRandomNumbers()
        drawnSamples = drawSamples(samples, randomNumbers)
        drawnLabels = drawLabels(labels, randomNumbers)
        j = 0
        sum = 0
        while j < len(drawnSamples):
            sum += gradient(drawnSamples, drawnLabels, theta) * \
                loss(drawnSamples, drawnLabels, theta)
            j += 1
        theta = theta - learningRate * 1/len(drawnSamples) * sum
        learningRate = learningRate / math.sqrt(t)
        t += 1


data = np.loadtxt('iris_data.csv', delimiter=',', usecols={0, 1, 2})
labels = np.loadtxt('iris_data.csv', delimiter=',',
                    usecols=3, dtype=str, converters={3: lambda s: convertLabel(s)})

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

colors = {0.: 'red', 1.: 'blue'}


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
# plt.show()
sample = data[0, :]
print(sample)

bla = hypothesis(sample, 1)
print(bla)
