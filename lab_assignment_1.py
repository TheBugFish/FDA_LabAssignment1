import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


class LogisticRegression:
    def __init__(self, iterations=100, sampleSize=20, learningRate=0.1):
        self.learningRate = learningRate
        self.iterations = iterations
        self.sampleSize = sampleSize

    def hypothesis(self, sampleset):
        z = np.dot(sampleset, self.theta)
        return 1 / (1 + np.exp(-z))

    def loss(self, sampleset, labelset, h):
        return (-labelset * np.log(h) - (1 - labelset) * np.log(1 - h)).mean()

    def gradient(self, sampleset, labelset, h):
        return np.dot(sampleset.T, (h - labelset))

    def getRandomNumbers(self, datalength):
        return random.sample(range(datalength), self.sampleSize)

    def drawSamples(self, samples, randomNumbers, featureCount):
        i = 0
        randomSample = np.zeros(shape=(len(randomNumbers), featureCount))
        while i < len(randomNumbers):
            randomSample[i] = samples[randomNumbers[i]]
            i += 1
        return randomSample

    def drawLabels(self, labels, randomNumbers):
        i = 0
        randomSample = np.zeros(shape=(len(randomNumbers), 1))
        while i < len(randomNumbers):
            randomSample[i] = labels[randomNumbers[i]]
            i += 1
        return randomSample

    def train(self, samples, labels):
        self.theta = np.zeros(samples.shape[1])
        print(self.theta)
        t = 0
        while t < self.iterations:
            randomNumbers = self.getRandomNumbers(len(samples))
            drawnSamples = self.drawSamples(samples, randomNumbers, 3)
            drawnLabels = self.drawLabels(labels, randomNumbers)
            h = self.hypothesis(drawnSamples)
            sum = self.gradient(drawnSamples, drawnLabels, h) / \
                self.loss(drawnSamples, drawnLabels, h)
            self.theta -= self.learningRate * (1/self.sampleSize) * sum
            self.learningRate = (self.learningRate / np.sqrt(t+1))
            t += 1


def convertLabel(l):
    labelString = l.decode()
    return 0 if labelString == 'Iris-setosa' else 1


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

plt.show()
model = LogisticRegression()
model.train(data, labels)
