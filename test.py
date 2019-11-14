import numpy as np
from matplotlib import pyplot as plt

l = [1,2,3,4]

# print(l)

def multiply_by_value(list, number):
        return [ i * number for i in list]

print(multiply_by_value(l, 3))

dict = {'apple': 3, 2.5 : 'pear', 'banana': 2 }

# print(dict[2.5])

array1 = np.array([1,2,3])
array2 = np.array([2,3,4])

# print(array1 * 3)
# print(array1 * array2)

M = np.array([
    [1,2],
    [3,4]
])
# print(M)

N = np.array([
    [2,3],
    [4,5]
])
# print(N)
# print(M @ N)

data = np.loadtxt('abalone.csv', skiprows=1, delimiter=',')
# print(data)

x = data[:,0]
y = data[:,1]

plt.scatter(x, y)
plt.xlabel('Diameter')
plt.ylabel('Weight')
plt.title('Abalone Diameter against Weight')
plt.show()




