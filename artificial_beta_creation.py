import numpy as np
import random as ra
from sklearn.preprocessing import normalize


def create_normal():
    a= [.8, 3, 1, 10]
    b = [10, 10, 10, .5]
    data = []
    for i in range(18500):
        temp = []
        
        if i >= 3000 and i <= 4000:
            if i%50 == 0:
                a[0] += .15
                a[1] +=  .15
                a[2] +=  .18
                a[3] -= .18
                #b[0] = b[0] - .2
                #b[1] -= .2
                #b[2] -= .12
                #b[3] += .12
        elif i >= 9000 and i <= 10000:
            if i%50 == 0:
                a[0] -= .15
                a[1] -=  .15
                a[2] -=  .18
                a[3] += .18
                #b[0] = b[0] + .2
                #b[1] += .2
                #b[2] += .12
                #b[3] -= .12
            
        elif i >=14000 and i < 16000:
            if i%50 == 0:
                a[0] += .1
                a[1] +=  .1
                a[2] +=  .06
                a[3] -= .06
               # b[0] = b[0] - .1
               # b[1] -= .1
               # b[2] -= .06
               # b[3] += .06
        elif i >= 17000 and i < 18000:
            if i%50 == 0:
                a[0] -= .1
                a[1] -=  .1
                a[2] -=  .06
                a[3] += .06
                #b[0] = b[0] + .1
                #b[1] += .1
                #b[2] += .06
                #b[3] -= .06
            
        
        temp.append(np.random.beta(a[0],b[0]))
        temp.append(np.random.beta(a[1],b[1]))
        temp.append(np.random.beta(a[2],b[2]))
        temp.append(np.random.beta(a[3],b[3]))
        data.append(temp)
            
    return data


def create_outliers():
    a= [.8, 4, 6, 7.5]
    b = [10, .5, .5, .5]
    data = []
    for i in range(2000):
        temp = []
        if i < 100:
            temp.append(np.random.beta(a[0],b[0]))
            temp.append(np.random.beta(a[1],b[1]))
            temp.append(np.random.beta(a[2],b[2]))
            temp.append(np.random.beta(a[3],b[3]))
            data.append(temp)
        elif i >= 100 and i <= 150:
            
            #if i%10 == 0:
            #    for j in range(len(b)):
            #        a[j] = a[j] + .25
            #    for j in range(len(b)):
            #        b[j] = b[j] + .01
            
            temp.append(np.random.beta(a[0],b[0]))
            temp.append(np.random.beta(a[1],b[1]))
            temp.append(np.random.beta(a[2],b[2]))
            temp.append(np.random.beta(a[3],b[3]))
            data.append(temp)
        elif i > 150 and i < 250:
            temp.append(np.random.beta(a[0],b[0]))
            temp.append(np.random.beta(a[1],b[1]))
            temp.append(np.random.beta(a[2],b[2]))
            temp.append(np.random.beta(a[3],b[3]))
            data.append(temp)
        elif i >= 250 and i <= 350:
            
            #if i%10 == 0:
             #   for j in range(len(b)):
            #        a[j] = a[j] - .01
             #   for j in range(len(b)):
             #       b[j] = b[j] - .01
            
            temp.append(np.random.beta(a[0],b[0]))
            temp.append(np.random.beta(a[1],b[1]))
            temp.append(np.random.beta(a[2],b[2]))
            temp.append(np.random.beta(a[3],b[3]))
            data.append(temp)
        else:
            temp.append(np.random.beta(a[0],b[0]))
            temp.append(np.random.beta(a[1],b[1]))
            temp.append(np.random.beta(a[2],b[2]))
            temp.append(np.random.beta(a[3],b[3]))
            data.append(temp)
    return data

            
if __name__ == '__main__':
    norm = create_normal()
    outliers = create_outliers()
    
    total = norm[0:500]
    norm = norm[500:]
    
    classes = [0 for i in range(500)]
    
    while len(norm) > 0 or len(outliers) > 0:
        if np.random.randint(0,9) == 8:
            if len(outliers)>0:
                total.append(outliers.pop(0))
                classes.append(1)
        else:
            if len(norm)>0:
                total.append(norm.pop(0))
                classes.append(0)
    #print(len(total))
    #total = normalize(total)
    np.savetxt('artificial_beta.csv', total)
    np.savetxt('artificial_beta_classes.csv', classes)
