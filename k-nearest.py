import math
import operator
import matplotlib.pyplot as plt
import numpy as np


def find_distance(point1, point2, length):
    distance = 0
    for x in range(length):
        distance += (point1[x] - point2[x])**2
    return math.sqrt(distance)


def find_color(neighbors, k):
    classes = {}
    for x in range(len(neighbors)):
        count = neighbors[x][-1]
        if count in classes:
            classes[count] += 1
        else:
            classes[count] = 1
    sorted_by_count = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)
    top_choice = int(sorted_by_count[0][0])
    return top_choice


def find_neighbors(training_set, test_point, k):
    distances = []
    length = len(test_point) - 1
    for x in range(len(training_set)):
        dist = find_distance(test_point, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    z_neighbors = []
    for x in range(k):
        z_neighbors.append(distances[x][0])
    return z_neighbors


print('k-nearest Neighbors Algorithm in Python')
var = int(input("Enter Value for K: "))
k = var
z = (k - 1) + k
filename = 'KNN_Data.csv'
np.genfromtxt(filename, delimiter=',')
dataset = np.genfromtxt(filename, delimiter=',')
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, -1])
testSet = np.array([[0.8, 0.2], [0.55, 0.2], [0.2, 00.4]])
plt.scatter(testSet[:, 0], testSet[:, 1])
print('Dataset')
print(dataset)
print('Test Points')
print(testSet)
plt.show()
predictions = []
print('Neighbors')
for x in range(len(testSet)):
        neighbors = find_neighbors(dataset, testSet[x], z)
        print(neighbors[0:k])
        color = find_color(neighbors, k)
        predictions.append(color)
predictions = np.asarray(predictions)
predictions = predictions.reshape((-1, 1))
newSet = np.concatenate((testSet, predictions), axis=1)
print('Final Results')
print(newSet)
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, -1])
plt.scatter(newSet[:, 0], newSet[:, 1], c=newSet[:, -1])
plt.show()
