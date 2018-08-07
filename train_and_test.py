import Autoencoder as ae
import numpy as np
import random as ra
import time
from scipy import stats
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt

'''Globals'''
n_epochs = 20
program_code = 0
file_name = ''
fixed_size = 0
def online_train(ann, data):
    print("Training Autoencoder")
    for i in range(n_epochs):
        print("Epoch " + str(i))
        for j in range(len(data)):
            rand = ra.randint(0, len(data)-1)
            ann.partial_fit([data[rand][:]])
    return ann


def calculate_threshold(window):
    std = np.std(window)
    mean = np.average(window)
    return mean+2*std

def z_test(window1, window2):
    if len(window1) == 0:
        return 0
    elif len(window2) == 0:
        return 0
    else:
        return ztest(window1, window2)

def get_data(FILE):
    data = []
    for line in open(FILE).readlines():
        temp = line.split(',')
        data.append(temp)
    return np.asarray(data[1:]).astype(float)


def get_class_labels(FILE):
    return np.genfromtxt(FILE)


def fix_train(data, classes):
    new_data = []
    for i in range(len(data)):
        if classes[i] == 0:
            new_data.append(data[i][:])
    return new_data

def test(ann, data, train_data):
    losses = []
    for i in range(len(data)):
        losses.append(ann.calc_total_cost([data[i][:]]))
    return losses


def base_test(ann, data,threshold, test_class, train_data):
    losses = []
    correct = 0
    accuracies = []
    predicted = []
    losses = []
    correct = 0
    accuracies = []
    predicted = []
    window = []
    window2 = []
    correct = 0
    anomaly = 0
    true_anomaly = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_precision = []
    total_recall = []
    precision = 0
    recall = 0
    total_f_measure = []
    losses = []
    total_alarms = []
    for i in range(len(data)):
        cost = ann.calc_total_cost([data[i][:]])
        losses.append(cost)
        if cost < threshold:
            p = 0
        else:
            p = 1
        predicted.append(p)
        if p == test_class[i]:
            correct +=1
        if p == 0 and test_class[i] == 0:
            true_negative +=1
        if p ==  1 and test_class[i] == 1:
            anomaly +=1
            true_positive +=1
        if p == 1 and test_class[i] == 0:
            false_positive +=1
        if p == 0 and test_class[i] == 1:
            false_negative +=1
        if test_class[i] == 1:
            true_anomaly +=1
        accuracies.append(correct/(i+1))
        if i %100 == 0:
            if true_positive + false_positive > 0 and i %100== 0:
                precision = true_positive/(true_positive + false_positive)
                total_precision.append(true_positive/(true_positive + false_positive))
                tp = True
            if true_positive + false_negative > 0 and i %100== 0:
                recall = true_positive/(true_positive + false_negative)
                total_recall.append(true_positive/(true_positive + false_negative))
                rec = True
        if i%100 == 0:
            if recall + precision > 0:
                total_f_measure.append(2*(recall*precision)/(recall+precision))
            else:
                if len(total_f_measure)>0:
                    recall = total_f_measure[len(total_f_measure)-1]
                else:
                    total_f_measure.append(0)
        if i %100 == 0:
            if false_positive + true_negative >0 and i %100 ==0:
                total_alarms.append(false_positive/(false_positive + true_negative))
    np.savetxt(str(file_name) + '_false_alarm_rate_baseline.csv', total_alarms)
    np.savetxt(str(file_name) + '_f_measure_baseline.csv', total_f_measure)
    np.savetxt(str(file_name) +'_precision_baseline.csv', total_precision)
    np.savetxt(str(file_name) +'_recall_baseline.csv', total_recall)
    np.savetxt(str(file_name) +'_accuracies_baseline.csv', accuracies)
    np.savetxt(str(file_name) +'_costs.csv',losses) 
    return losses, predicted



def window_test(ann, data, threshold, test_class, window3):
    losses = []
    correct = 0
    accuracies = []
    predicted = []
    window = []
    window2 = []
    correct = 0
    anomaly = 0
    true_anomaly = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_precision = []
    total_recall = []
    precision = 0
    recall = 0
    total_f_measure = []
    total_alarms = []
    losses = []
    data_window = []
    for i in range(len(data)):
        cost = ann.calc_total_cost([data[i][:]])
        losses.append(cost)
        window2.append(cost)
        if len(window2) >50:
            window2.pop(0)
        if len(window) > 50:
            window.append(cost)
        else:
            window.append(cost)
        if len(window3) < 50:
            window3.append(cost)
        if len(window2) >= 50:
            if z_test(window3, window2)[1] < .05:
                window = window[len(window)-50:len(window)]
                window3 = window2
                window2 = []
                threshold = calculate_threshold(window)
        if len(window)%100 == 0:
            threshold = calculate_threshold(window)
        if len(window) == 1000:
            window.pop(0)
            p = 0
        else:
            p = 1
        predicted.append(p)
        if p == test_class[i]:
            correct +=1
        if p == 0 and test_class[i] == 0:
            true_negative +=1
        if p ==  1 and test_class[i] == 1:
            anomaly +=1
            true_positive +=1
        if p == 1 and test_class[i] == 0:
            false_positive +=1
        if p == 0 and test_class[i] == 1:
            false_negative +=1
        if test_class[i] == 1:
            true_anomaly +=1
        accuracies.append(correct/(i+1))
        if true_positive + false_positive > 0 and i%100 == 0:
            precision = true_positive/(true_positive + false_positive)
            total_precision.append(true_positive/(true_positive + false_positive))
        if true_positive + false_negative > 0 and i%100 ==0:
            recall = true_positive/(true_positive + false_negative)
            total_recall.append(true_positive/(true_positive + false_negative))
        if i%100 == 0:
            if precision +recall > 0:
                total_f_measure.append(2*(recall*precision)/(recall+precision))
        if i %100 == 0:
            if false_positive + true_negative >0 and i %100 == 0:
                total_alarms.append(false_positive/(false_positive + true_negative))
    np.savetxt(str(file_name) +'_false_alarm_window.csv', total_alarms)
    np.savetxt(str(file_name) +'_f_measure.csv', total_f_measure)
    np.savetxt(str(file_name) +'_precision_window.csv', total_precision)
    np.savetxt(str(file_name) +'_recall_window.csv', total_recall)
    np.savetxt(str(file_name) +'_accuracies_window.csv', accuracies)
    return losses, predicted

def fixed_window(ann, data, threshold, test_class, window3):
    global fixed_size
    losses = []
    correct = 0
    accuracies = []
    predicted = []
    window = []
    window2 = []
    correct = 0
    anomaly = 0
    true_anomaly = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_precision = []
    total_recall = []
    precision = 0
    recall = 0
    total_f_measure = []
    losses = []
    data_window = []
    total_alarms = []
    print('What Size Fixed Window would you like?')
    fixed_size = int(input())
    print('Fixed Size is '+ str(fixed_size))
    
    for i in range(len(data)):
        cost = ann.calc_total_cost([data[i][:]])
        
        losses.append(cost)
        window.append(cost)
        if len(window) == fixed_size:
            threshold = calculate_threshold(window)
            window = []
        if cost < threshold:
            p = 0
        else:
             p= 1
        predicted.append(p)
        if p == test_class[i]:
            correct +=1
        if p == 0 and test_class[i] == 0:
            true_negative +=1
        if p ==  1 and test_class[i] == 1:
            anomaly +=1
            true_positive +=1
        if p == 1 and test_class[i] == 0:
            false_positive +=1
        if p == 0 and test_class[i] == 1:
            false_negative +=1
        if test_class[i] == 1:
            true_anomaly +=1
        accuracies.append(correct/(i+1))
        
        if true_positive + false_positive > 0 and i %100 == 0:
            precision = true_positive/(true_positive + false_positive)
            total_precision.append(true_positive/(true_positive + false_positive))
        if true_positive + false_negative > 0 and i %100 == 0:
            recall = true_positive/(true_positive + false_negative)
            total_recall.append(true_positive/(true_positive + false_negative))
        if i%100 == 0:
            if precision +recall > 0:
                total_f_measure.append(2*(recall*precision)/(recall+precision))
        if i %100 == 0:
            if false_positive + true_negative >0 and i%100 ==0:
                total_alarms.append(false_positive/(false_positive + true_negative))
    
    np.savetxt(str(file_name) +'_false_alarm_fixed_window_'+str(fixed_size) +'.csv', total_alarms)
    np.savetxt(str(file_name) +'_f_measure_fixed_window_'+str(fixed_size) +'.csv', total_f_measure)
    np.savetxt(str(file_name) +'_precision_fixed_window_'+str(fixed_size) +'.csv', total_precision)
    np.savetxt(str(file_name) +'_recall_fixed_window_'+str(fixed_size) +'.csv', total_recall)
    np.savetxt(str(file_name) +'_accuracies_fixed_window_'+str(fixed_size) +'.csv', accuracies)
    return losses, predicted

    


if __name__ == '__main__':
    print('Enter the File you are Inputing')
    file_name = input()
    data = get_data(file_name)
    file_name = file_name[0:len(file_name)-4]
    test_class = data[:,len(data[0])-1]
    test_class = test_class[500:]
    data = np.delete(data, -1, axis = 1)
    train_data = data[0:500]
    test_data = data[500:]
    print('Enter 1 if you want to do Baseline test')
    print('Enter 2 if you want to do Adaptive Window Test')
    print('Enter 3 if you want to do Fixed Window Test')
    program_choice = int(input())

    
    n_features = len(train_data[0])
    n_hidden = int(n_features*.5)
   
    ann = online_train(ae.Autoencoder(n_features, n_hidden), train_data)
    validation = train_data
    val_loss = test(ann, validation, train_data)
    threshold = calculate_threshold(val_loss)

    
    if program_choice == 1:
        losses,predicted = base_test(ann, test_data, threshold, test_class, train_data)
    elif program_choice == 2:
        losses,predicted = window_test(ann, test_data, threshold, test_class, val_loss)
    elif program_choice == 3:
        losses,predicted = fixed_window(ann, test_data, threshold, test_class, val_loss)
    else:
        print('Entered Incorrect program choice')
        exit()
    
    correct = 0
    anomaly = 0
    true_anomaly = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(test_class)):
        if predicted[i] == test_class[i]:
            correct +=1
        if predicted[i] == 0 and test_class[i] == 0:
            true_negative +=1
        if predicted[i] == 1 and test_class[i] == 1:
            true_positive +=1
            anomaly +=1
        if predicted[i] == 1 and test_class[i] == 0:
            false_positive +=1
        if predicted[i] == 0 and test_class[i] == 1:
            false_negative +=1
        if test_class[i] == 1:
            true_anomaly +=1
    if true_positive + false_positive == 0:
        precision = 0
    else:         
        precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    print("Accuracy " + str(correct/len(predicted))) 
    print("Detection Rate " + str(anomaly/true_anomaly))
    if program_choice == 1:
        f = open(str(file_name) +'_baseline_metrics.txt', 'w')
    elif program_choice == 2:
        f = open(str(file_name) +'_adaptive_window_metrics.txt', 'w')
    elif program_choice == 3:
        f = open(str(file_name) +'_fixed_' +str(fixed_size) + '_metrics.txt', 'w')
    f.write("Accuracy  " + str(correct/len(predicted)) + '\n')
    f.write("Precision " + str(true_positive/(true_positive+false_positive)) +'\n')
    f.write("Recall " + str(true_positive/(true_positive+false_negative))+'\n')
    f.write("Detection Rate " + str(anomaly/true_anomaly)+'\n')
    f.write("Correct Detection Number " + str(true_positive) +'\n')
    f.write("False Alarm Number " + str(false_positive) + '\n')
    f_measure = 2*((precision*recall)/(precision+recall))
    f.write("F Measure " + str(f_measure)+'\n')
    f.close()
   
