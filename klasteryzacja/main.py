import pandas as pd
import seaborn as sb
import random


class DataProcessing:
    data = None

    def __init__(self, d):
        self.data = d
        pd.set_option('display.max_rows', len(self.data))

    def przetasowanie(self):
        for i in range(len(self.data) - 1, 0, -1):
            j = random.randint(0, i)
            self.data.iloc[i], self.data.iloc[j] = self.data.iloc[j], self.data.iloc[i]
        return self.data

    def normalizacja(self):
        listCopy = self.data.copy()
        wartosci = listCopy.select_dtypes(exclude="object")  # no variety column
        nazwy_kolumn = wartosci.columns.tolist()
        for kolumna in nazwy_kolumn:
            data = listCopy.loc[:, kolumna]
            minimum, maximum = min(data), max(data)
            for rzad in range(0, len(listCopy), 1):
                value = (listCopy.at[rzad, kolumna] - minimum) / (maximum - minimum)  # (x-min)/(max-min)
                listCopy.at[rzad, kolumna] = value
        self.data = listCopy

    def podzial(self):
        split = int(len(self.data) * 0.7)
        lista_1 = self.data.iloc[:split, :]
        lista_2 = self.data.iloc[split:, :]
        return lista_1, lista_2


class KNN:
    @staticmethod
    def minkowskiDistance(vec1, vec2, m):
        odleglosc = 0
        for i in range(len(vec2) - 1):
            odleglosc += abs(vec2[i] - vec1[i]) ** m
        odleglosc = float(odleglosc ** (1 / m))
        return odleglosc

    @staticmethod
    def sortByDistance(testIris, trainBase, m):
        odleglosci = []
        for i in range(len(trainBase)):
            odleglosci.append(KNN.minkowskiDistance(testIris, trainBase.iloc[i], m))
        return KNN.sorting(trainBase, odleglosci)

    @staticmethod
    def sorting(trainList, odleglosci):
        trainBase = trainList.copy()
        for i in range(len(trainBase)):
            gotowy = True
            print(".", end="")
            for j in range(len(trainBase) - i - 1):
                if odleglosci[j] > odleglosci[j + 1]:
                    odleglosci[j], odleglosci[j + 1] = odleglosci[j + 1], odleglosci[j]
                    trainBase.iloc[j], trainBase.iloc[j + 1] = trainBase.iloc[j + 1], trainBase.iloc[j]
                    gotowy = False
            if gotowy:
                break
        return trainBase

    @staticmethod
    def clustering(valBase, trainBase, k, m):
        poprwiony = 0
        n = len(valBase)
        blad = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
        typy = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
        for irisId in range(n):
            testIris = valBase.iloc[irisId]
            testIrisVariety = valBase.iloc[irisId].variety
            typy[testIrisVariety] += 1

            klasy = {"Setosa": 0, "Virginica": 0, "Versicolor": 0}
            print("\n({}/{})".format(irisId + 1, n))

            trainBase = KNN.sortByDistance(testIris, trainBase, m)

            for i in range(0, k, 1):
                klasy[trainBase.iloc[i].variety] += 1

            aiIris = max(klasy, key=klasy.get)
            if aiIris == testIrisVariety:
                poprwiony += 1
            else:
                blad[testIrisVariety] += 1

        print("\n\tResults:\n"
              "k = {}\nm = {}".format(k, m))
        k = 0
        for klucz, wartosc in typy.items():
            all = list(typy.values())[k]
            this = list(blad.values())[k]
            diff = all - this
            print("{} - {}/{} - {:.2f}%".format(klucz, diff, all, diff / all * 100))
            k += 1
        accuracy = poprwiony / len(valBase) * 100
        print("Accuracy - {:.2f}%".format(accuracy))
        return accuracy


dataSet = pd.read_csv('iris.csv')
dp = DataProcessing(dataSet)
acc = []
for k in range(2, 6, 1):
    dp.przetasowanie()
    dp.normalizacja()
    training, values = dp.podzial()
    acc.append(KNN.clustering(values, training, k, 2))
for k in range(2, 6, 1):
    print("k = {} - {:.2f}% accuracy.".format(k, acc[k - 2]))




