# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def getDataset():
    data = pd.read_csv("mushroom-attributions-200-samples.csv")
    data = data[['odor', 'bruises']]  # Reduce dims of dataset in order to visualize
    dataset = []
    l = len(data)
    # l = 100
    for i in range(l):
        dataset.append(data.loc[i])
    return dataset


def generate2dData():
    xs = []
    ys = []
    n = 500
    for i in range(n):
        xs.append(random.random())
        ys.append(random.random())
        if i > .9 * n:
            xs.append(random.random() + .5)
            ys.append(random.random() + .5)
    data = []
    for i in range(n):
        data.append([xs[i], ys[i]])
    return data


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


def test1D():
    data = []
    for i in range(3):
        data.append([i])
        data.append([i])
        data.append([i + 10])
        data.append([i + 10])
    main(data)


def test2D():
    main(generate2dData())


def testMushroomDataset():
    df = pd.read_csv('mushroom-attributions-200-samples.csv')
    dataset = []
    for i in range(len(df)):
        dataset.append(df.loc[i])
    main(dataset)


def plotClusters(clusters, data):
    for i in clusters[0]:
        plt.scatter(data[i][0], data[i][1], c='red')
    for i in clusters[1]:
        plt.scatter(data[i][0], data[i][1], c='green')


# This method checks whether there exists some data points in different clusters
def checkClustersContainDifferent(clusters):
    for i in range(len(clusters)):
        for d in clusters[i]:
            for j in range(len(clusters)):
                if i != j:
                    if d in clusters[j]:
                        return False
    return True


def standardizeData(df):
    features = df.columns
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target

    # Standardizing the features
    x = StandardScaler().fit_transform(x)


def pcaDataAndPlot(x, clusters):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    colors = ['blue', 'black', 'yellow', 'green', 'purple']
    plt.figure(figsize=(10, 8))
    plt.xlim(-3, 3)
    for i in clusters[0]:
        plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[0])
    for i in clusters[1]:
        plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[1])
    for i in clusters[2]:
        plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[2])
    for i in clusters[3]:
        plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[3])
    for i in clusters[4]:
        plt.scatter(principalDf.loc[i][0], principalDf.loc[i][1], c=colors[4])


def main(data):
    n = len(data)
    numOfLoops = 100
    k = 5
    m = 12
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

    if len(data[0]) <= 2:
        plotClusters(clusters, data)

    print(clusters)
    print(checkClustersContainDifferent(clusters))


if __name__ == '__main__':
    # testList = [[9], [11], [9], [8], [20], [7], [4]]
    # testPrintTables(testList)

    # data = getDataset()
    # main(data)

    # test1D()

    # test2D()

    testMushroomDataset()
