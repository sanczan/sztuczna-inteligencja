import numpy as np
import pandas as pd
import random
import math

class DataProcessing:
    @staticmethod
    def przetasuj(X):
        for i in range(len(X)-1,0,-1):
            j = random.randint(0,i)
            X.iloc[i], X.iloc[j] = X.iloc[j], X.iloc[i]
        return X
    @staticmethod
    def normalizacja(n):
        k = n.drop("variety", axis=1)
        k1 = (k - k.min())/(k.max()-k.min())
        for (name,col) in k1.items():
            n[name] = col
        return n
    @staticmethod
    def podzial(X, x):
        if x >= 10 or x < 0:
            raise ValueError('Niepoprawne x (musi byæ < 1)')
        return X[:math.ceil(len(X)*x)],X[math.ceil(len(X)*(1-x)):]

class bayes:
    def trojkat(x, srednia, odchyl):
        lewo = srednia - math.sqrt(6) * odchyl
        prawo = srednia + math.sqrt(6) * odchyl
        if x < lewo or x > prawo:
            return 0
        if x <= srednia:
            return (x - lewo) / ((srednia - lewo) * odchyl)
        else:
            return (prawo - x) / ((prawo - srednia) * odchyl)

    def classify(trening, probka):
        imiona = trening.variety.unique()
        classes = []
        for imie in imiona:
            classes += [trening[trening['variety'] == imie]]
            del classes[-1]['variety']
        classes_gauss = []
        for classy in classes:
            srednia = []
            odchyl = []
            gauss = []
            for (imie, data) in classy.items():
                srednia += [np.mean(data.values)]
                odchyl += [np.std(data.values)]
                gauss += [bayes.trojkat(probka[imie], srednia[-1], odchyl[-1])]
            classes_gauss += [math.prod(gauss)]
        return imiona[classes_gauss.index(max(classes_gauss))]


irysy = pd.read_csv('iris.csv')
irysy = DataProcessing.przetasuj(irysy)
irysy = DataProcessing.normalizacja(irysy)
irysy_treningowa, irysy_testowa = DataProcessing.podzial(irysy, 0.7)
poprawne = 0
for i in range(0,len(irysy_testowa)):
    probka = irysy_testowa.iloc[i].drop('variety').to_dict()
    if irysy_testowa.iloc[i].variety == bayes.classify(irysy_treningowa, probka):
        poprawne += 1
dokladnosc = poprawne / len(irysy_testowa.index) * 100
print("Dokladnosc (znormalizowana) -", dokladnosc, "%")