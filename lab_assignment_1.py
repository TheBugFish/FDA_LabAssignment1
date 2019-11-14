import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def convertLabel(l):
    labelString = l.decode()
    return 0 if labelString == 'Iris-setosa' else 1


data = np.loadtxt('iris_data.csv', delimiter=',', usecols={0, 1, 2})
labels = np.loadtxt('iris_data.csv', delimiter=',',
                    usecols=3, dtype=str, converters={3: lambda s: convertLabel(s)})

print(data)
print(labels)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

colors = {0.: 'red', 1.: 'blue'}


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
plt.show()
