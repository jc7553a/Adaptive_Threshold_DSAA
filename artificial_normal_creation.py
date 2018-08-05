import numpy as np
import random as ra
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from sklearn.preprocessing import normalize

def create_norm():
    data = []
    mu1 = 1
    mu2 = 5
    mu3 = -1
    mu4 = -7
    sig1 = .25
    sig2 = .5
    sig3 = .25
    sig4 = .1
    for i in range(20500):
        temp = []
        if i >=3500 and i < 7500 :
            if i%60 == 0:
                mu1 +=.1
                mu2 +=.1
                mu3 -=.1
                mu4 -=.1
        if i >= 7500 and i < 11500:
            if i%60 == 0:
                mu1 -=.1
                mu2 -=.1
                mu3 +=.1
                mu4 +=.1
        if i >= 15000 and i < 16500:
            if i%100 == 0:
                mu1 +=.25
                mu2 +=.25
                mu3 -=.25
                mu4 -=.25
        if i >= 17500 and i < 19000:
            if i%100 == 0:
                mu1 -=.25
                mu2 -=.25
                mu3 +=.25
                mu4 +=.25
                
        temp.append(ra.gauss(mu1, sig1))
        temp.append(ra.gauss(mu2, sig2))
        temp.append(ra.gauss(mu3, sig3))
        temp.append(ra.gauss(mu4, sig4))
        data.append(temp)
    return data

def create_outliers():
    for i in range(2000):
        data = []
    mu1 = 5
    mu2 = 9
    mu3 = -3
    mu4 = -10
    sig1 = .25
    sig2 = .1
    sig3 = .25
    sig4 = .25
    for i in range(2000):
        temp = []
        if i >=300 and i < 500 :
            if i%3 == 0:
                mu1 +=.1
                mu2 +=.1
                mu3 -=.1
                mu4 -=.1
        if i >= 700 and i < 900:
            if i%3 == 0:
                mu1 -=.1
                mu2 -=.1
                mu3 +=.1
                mu4 +=.1
        if i >= 1500 and i < 1600:
            if i%10 == 0:
                mu1 +=.25
                mu2 +=.25
                mu3 -=.25
                mu4 -=.25
        if i >= 1700 and i < 1900:
            if i%10 == 0:
                mu1 -=.25
                mu2 -=.25
                mu3 +=.25
                mu4 +=.25
                
        temp.append(ra.gauss(mu1, sig1))
        temp.append(ra.gauss(mu2, sig2))
        temp.append(ra.gauss(mu3, sig3))
        temp.append(ra.gauss(mu4, sig4))
        data.append(temp)
    return data


def create_full(norm, outlier):
    total = norm[0:500]
    norm = norm[500:]
    labels = []
    while len(norm) > 0 or len(outlier) >0:
        if ra.randint(0,10) == 8:
            if len(outlier) > 0:
                total.append(outlier.pop(0))
                labels.append(1)
        else:
            if len(norm) > 0:
                total.append(norm.pop(0))
                labels.append(0)
    return total, labels
            
if __name__ == '__main__':
    norm = create_norm()
    outlier = create_outliers()
    data, labels = create_full(norm, outlier)
    file_name = 'artificial_normal_distribution.csv'
    data = normalize(data)
    np.savetxt(file_name, data)
    np.savetxt('labels.csv', labels)
