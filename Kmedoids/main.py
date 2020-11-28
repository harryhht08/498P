# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import random
import pandas as pd
import matplotlib.pyplot as plt


# The only missing method is the portal where we can get the data

def getDataset():
    data = pd.read_csv("mushroom-attributions-200-samples.csv")
    data = data[['odor', 'bruises']]  # Reduce dims of dataset in order to visualize
    dataset = []
    l = len(data)
    # l = 100
    for i in range(l):
        dataset.append(data.loc[i])
    return dataset


# data: the actual data we use for this algorithm, it is a collection of data points, each point has dimension dims
def computeDistance(i, j, data):
    dims = len(data[0])
    dis = 0
    for x in range(dims):
        dis += (data[i][x] - data[j][x]) ** 2
    return dis ** .5


def buildRankTable(data):
    n = len(data)
    rank = [[(0, 0) for i in range(n)] for j in range(n)]
    similarityMetrix = [[(0, 0) for i in range(n)] for j in range(n)]

    # Compute the distance and record them in a tuple
    for i in range(n):
        for j in range(i, n):
            d = computeDistance(i, j, data)
            rank[i][j] = (d, j)
            rank[j][i] = (d, i)

    for i in range(n):
        rank[i].sort(key=lambda x: x[0])  # Sort based on distances
        r = 0
        rankings = []
        for j in range(n):
            rankings.append(r)
            if j <= n - 2 and rank[i][j][0] == rank[i][j + 1][0]:
                r -= 1
            r += 1
            similarityMetrix[i][j] = rank[i][j][1]  # Store the sample points
            rank[i][j] = rank[i][j][1]

        newList = [0] * n
        for j in range(n):
            newList[rank[i][j]] = rankings[j]

        rank[i] = newList

    return rank, similarityMetrix


# tableS: a nxn matrix containing all Sij values
# setG: set of integers - indexes of data points
# n: size of dataset
def getHv(index, m, n, rankTable, setG):
    hv = m * (m + 1) / 2
    for j in range(n):
        if not j in setG:
            for r in rankTable[index]:
                if r < rankTable[index][j]:
                    hv += 1
    return hv


def randomMedoids(k, data):
    medoids = set()
    for i in range(k):
        oldLen = len(medoids)
        while (len(medoids) == oldLen):
            medoids.add(random.randint(0, len(data) - 1))
    return medoids


def assignToClusters(k, n, medoids, rankTable):
    groups = []  # list of Set
    for i in range(k):
        groups.append([])  # Make k empty groups
    medoidsList = list(medoids)

    for i in range(n):
        rankRowI = rankTable[i]
        min = n
        for j in range(k):
            m = medoidsList[j]
            if rankRowI[m] < min:
                min = rankRowI[m]
        for j in range(k):
            m = medoidsList[j]
            if rankRowI[m] == min:
                groups[j].append(i)

        # cursor = 0
        # while (cursor < n):
        #     if similarityMetrix[i][cursor] in medoids:
        #         for groupIndex in range(k):
        #             if similarityMetrix[i][cursor] == medoidsList[groupIndex]:
        #                 groups[groupIndex].append(similarityMetrix[i][cursor])
        #         break  # Break the while loop if found the closest medoid
        #     cursor += 1
    return groups


def updateMedoids(k, m, n, medoids, similarityTable, rankTable):
    newMedoids = set()
    for med in medoids:
        mostSimilar = similarityTable[med][:m]
        maxHv = (-1, -1)
        for simi in mostSimilar:
            hv = getHv(simi, m, n, rankTable, set(mostSimilar))
            if hv > maxHv[1]:
                maxHv = (simi, hv)
        newMedoids.add(maxHv[0])


def printNice(table):
    for i in table:
        print(i)


def testPrintTables(data):
    r, s = buildRankTable(data)
    print("Rank Table: ")
    printNice(r)
    print()
    print("Similarity Table: ")
    printNice(s)


def test01():
    data = []
    for i in range(3):
        data.append([i])
        data.append([i])
        data.append([i + 10])
        data.append([i + 10])
    main(data)


def main(data):
    n = len(data)
    numOfLoops = 5000
    k = 2
    m = 3
    medoids = randomMedoids(k, data)
    rankTable, similarityMetrix = buildRankTable(data)
    for i in range(numOfLoops):
        updateMedoids(k, m, n, medoids, similarityMetrix, rankTable)
    clusters = assignToClusters(k, n, medoids, rankTable)

    # groups = assignToClusters(k, n, medoids, rankTable)
    # newMedoids = set()
    # for group in groups:
    #
    #     # The first position stores the new medoid, the second stores the maximum hv value
    #     maxHv = (-1, -1)
    #     for point in group:
    #         hv = getHv(point, m, n, rankTable, group)
    #         if hv > maxHv[1]:
    #             maxHv = (point, hv)
    #     newMedoids.add(maxHv[0])
    # medoids = newMedoids
    # Output: Groups!
    print('sss')


if __name__ == '__main__':
    # testList = [[9], [11], [9], [8], [20], [7], [4]]
    # testPrintTables(testList)

    # data = getDataset()
    # main(data)

    test01()

    print("hello")
