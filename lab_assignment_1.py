import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random


def convertLabel(l):
    labelString = l.decode()
    return 0 if labelString == 'Iris-setosa' else 1


def hypothesis(sample):
    res = 0.
    for i in sample:
        res += (1 / (1 + math.exp(-i)))
    return res


def loss(sampleset, labelset, theta):
    result = 0
    index = 0
    for i in sampleset:
        result += (-labelset[index] * math.log(hypothesis(i))) - (
            1 - labelset[index]) * math.log(1 - hypothesis(i))
        index += 1
    print(result)
    return result


def gradient(sampleset, labelset, theta):
    result = 0
    index = 0
    for i in sampleset:
        result += ((hypothesis(i) - labelset[index]) * i)
    return result


def getRandomNumbers(sampleAmount, datalength):
    return random.sample(range(datalength), sampleAmount)


def drawSamples(samples, randomNumbers, featureCount):
    i = 0
    randomSample = np.zeros(shape=(len(randomNumbers), featureCount))
    while i < len(randomNumbers):
        randomSample[i] = samples[randomNumbers[i]]
        i += 1
    return randomSample


def drawLabels(labels, randomNumbers):
    i = 0
    randomSample = np.zeros(shape=(len(randomNumbers), 1))
    while i < len(randomNumbers):
        randomSample[i] = labels[randomNumbers[i]]
        i += 1
    return randomSample


def SGD(samples, labels):
    theta = 0
    T = 100
    t = 0
    learningRate = 0.1
    sampleAmount = 20
    while t < T:
        randomNumbers = getRandomNumbers(sampleAmount, len(samples))
        drawnSamples = drawSamples(samples, randomNumbers, 3)
        drawnLabels = drawLabels(labels, randomNumbers)
        j = 0
        sum = 0
        test = loss(drawnSamples, drawnLabels, theta)
        print(test)
        while j < len(drawnSamples):
            sum += (gradient(drawnSamples, drawnLabels, theta)
                    * loss(drawnSamples, drawnLabels, theta))

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
ax = fig.add_subplot(111, projection='3d')
idx = 0
while idx < len(x):
    if int(labels[idx]) == 0:
        ax.scatter(x[idx], y[idx], z[idx], color=colors[0])
    else:
        ax.scatter(x[idx], y[idx], z[idx], color=colors[1])
    idx += 1

# plt.show()
SGD(data, labels)
