# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:16:29 2023

@author: Haoyi Zheng
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.stats
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import balanced_accuracy_score
os.chdir(r'.\fNIRS Data Collection\cutin')

def EMA_filter(signal, alpha):
    filtered_signal = np.zeros(signal.shape)
    for i in range(len(signal)):
        if i == 0:
            filtered_signal[i] = signal[i]
        else:
            filtered_signal[i] = (1-alpha)*filtered_signal[i-1]+alpha*signal[i]
    return filtered_signal

def EMA_filter_8ch(signals, alpha):
    filtered_signals=np.zeros(signals.shape)
    for i in range(len(signals)):
        filtered_signals[i,:] = EMA_filter(signals[i],alpha)
    return filtered_signals

def DEMA_filter_8ch(signals,alpha):
# Double Exponential Moving Average
    ema_signals=np.zeros(signals.shape)
    ema_ema_signals=np.zeros(signals.shape)
    for i in range(len(signals)):
        ema_signals[i,:] = EMA_filter(signals[i],alpha)
        ema_ema_signals[i,:] = EMA_filter(ema_signals[i],alpha)
    dema_signals = 2*ema_signals - ema_ema_signals
    return dema_signals

def MA_smoothing(signals, windowlen):
    smoothed_signals=np.zeros(signals.shape)
    # print(signals[0].shape)
    for i in range(len(signals)):
        padded_signal=np.pad(signals[i][:-1],(int(windowlen/2),int(windowlen/2)),mode='edge')
        # print(len(padded_signal))
        smoothed_signals[i,:] = np.convolve(padded_signal,np.ones(windowlen)/windowlen,mode='valid')
    return  smoothed_signals

def get_fnirs_features(fnirs_data, waveletname):
    list_features = []
    ######fnirs_data:400, waveletname:'db4'
    list_coeff = pywt.wavedec(fnirs_data, waveletname)    #6
    features = []
    for coeff in list_coeff:
        # print(coeff)
        features += get_features(coeff)
    # print('features', np.array(features).shape)
    list_features.append(features)                        #72
    return list_features

def calculate_entropy(list_values):     #1 feature
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):      #9 features
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]


def calculate_crossings(list_values):       #2 features
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


def get_features(list_values):

    entropy = calculate_entropy(list_values)
    # print('entropy', np.array(entropy).shape)
    crossings = calculate_crossings(list_values)
    # print('crossings',np.array(crossings).shape)
    statistics = calculate_statistics(list_values)
    # print('statistics',np.array(statistics).shape)
    return [entropy] + crossings + statistics

def select_IG(feature, label):
    # x, y = load_iris(return_X_y=True)
    importances = mutual_info_classif(feature, label,random_state=2)
    # print(feature)
    # print(label)
    # print(importances)
    feature_label = np.arange(0, feature.shape[1])
    feat_importances = pd.Series(importances, feature_label)
    index_impo = np.argsort(importances)
    # print(index_impo)
    # feat_importances.plot(kind='barh', color='teal')
    # plt.savefig('./mutual_info_classif.png')
    # plt.show()
    # plt.close()
    return index_impo,feat_importances

feature_list_total=[]
label_list_total=[]
for NUM in range(1,11):
    HHb=np.load('fNIRS_total_HHb{}.npy'.format(NUM),allow_pickle=True)
    O2Hb=np.load('fNIRS_total_O2Hb{}.npy'.format(NUM),allow_pickle=True)
    human=np.load('human_label_total{}.npy'.format(NUM),allow_pickle=True)


    fNIRS_COE_list=[]
    human_list=[]
    for i in range(len(human)):
        if len(O2Hb[i]) != 0:
            for j in range(len(O2Hb[i])):
                if j%5==0:
                    fNIRS_window_O2Hb=O2Hb[i][j]
                    fNIRS_window_HHb = HHb[i][j]
                    fNIRS_filtered_O2Hb_short = DEMA_filter_8ch(fNIRS_window_O2Hb, 0.06)
                    fNIRS_filtered_O2Hb_long = DEMA_filter_8ch(fNIRS_window_O2Hb, 0.001)
                    fNIRS_filtered_O2Hb = fNIRS_filtered_O2Hb_short - fNIRS_filtered_O2Hb_long
                    fNIRS_filtered_O2Hb = MA_smoothing(fNIRS_filtered_O2Hb, 100)  # 觉得滤波滤不干净的话可以考虑加上平滑？
                    # 滤波 先用带通EMA再移动平均
                    fNIRS_filtered_HHb_short = DEMA_filter_8ch(fNIRS_window_HHb, 0.06)
                    fNIRS_filtered_HHb_long = DEMA_filter_8ch(fNIRS_window_HHb, 0.001)
                    fNIRS_filtered_HHb = fNIRS_filtered_HHb_short - fNIRS_filtered_HHb_long
                    fNIRS_filtered_HHb = MA_smoothing(fNIRS_filtered_HHb, 100)  # 真的需要吗？

                    fNIRS_window_COE = (fNIRS_filtered_O2Hb - fNIRS_filtered_HHb) / 2 ** 0.5
                    fNIRS_COE_list.append(fNIRS_window_COE[:,-200:])
                    human_list.append(human[i][j])
    fNIRS_COE_array=np.array(fNIRS_COE_list)
    label_list=np.array(human_list)

    print('Extracting Features Subject S{}'.format(NUM))
    time.sleep(1)
    features_list=[]
    index_matrix=[]
    for i in tqdm(range(8)):
        features_list_1ch=[]
        for j in range(len(fNIRS_COE_array)):
            fNIRS_window_COE = fNIRS_COE_array[j,i,:]
            [features]=get_fnirs_features(fNIRS_window_COE,'db4')
            features_list_1ch.append(np.array(features))
        features_1ch=np.vstack(features_list_1ch)
        index_impo, _ = select_IG(features_1ch, label_list)
        # print(index_impo[-4:])
        index_matrix.append(index_impo[-4:])
        features_list.append(features_1ch[:,index_impo[-4:]].T)
    features_list=np.array(features_list)
    index_matrix=np.array(index_matrix)
    features_list_reshape=[]
    for i in range(features_list.shape[2]):
        features_list_reshape.append(features_list[:,:,i].ravel())
    features_select=np.array(features_list_reshape)
    feature_list_total.append(features_select)
    label_list_total.append(label_list)

# feature_array_total=np.concatenate(feature_list_total,axis=0)
# label_array_total=np.concatenate(label_list_total,axis=0)
# print(feature_array_total.shape,feature_array_total.shape)


# start_index=np.random.randint(0,427)#89
# print(start_index)
# end_index=start_index+200
# slices = np.r_[:start_index, end_index:len(label_list)]
random_seed=1

# class_weights = {0: 0.3, 1: 1}
ada_clf_tree = AdaBoostClassifier(
    # SVC(kernel='linear'), n_estimators=200,
    DecisionTreeClassifier(max_depth=5,criterion='entropy'), n_estimators=300,
    # algorithm='SAMME', learning_rate=0.5
    algorithm='SAMME.R', learning_rate=0.1, random_state=random_seed
    )

ada_clf_svm = AdaBoostClassifier(
    SVC(kernel='linear'), n_estimators=1000,
    # algorithm='SAMME', learning_rate=0.5
    algorithm='SAMME', learning_rate=0.1, random_state=random_seed
    )

ada_clf_gaussian = AdaBoostClassifier(
    GaussianNB(), n_estimators=1000,
    # algorithm='SAMME', learning_rate=0.5
    algorithm='SAMME.R', learning_rate=0.1, random_state=random_seed
    )
#
ada_clf_logistic = AdaBoostClassifier(
    LogisticRegression(max_iter=1000), n_estimators=1000,
    # algorithm='SAMME', learning_rate=0.5
    algorithm='SAMME.R', learning_rate=0.1, random_state=random_seed
    )

forest_clf = RandomForestClassifier(n_estimators=400)

mplc_clf = MLPClassifier(hidden_layer_sizes=(100,100,50 ), max_iter=3500,batch_size=32)

voting_clf = VotingClassifier(estimators=[('DecisionTree_Adaboost',ada_clf_tree),
                                          ('SupportingVector_Adaboost',ada_clf_svm),
                                          ('NaiveBayesian_Adaboost',ada_clf_gaussian),
                                          ('LogisticRegress_Adaboost',ada_clf_logistic),
                                          ('RandomForest',forest_clf),
                                          ('MultiLayerPerceptron',mplc_clf)
                                          ],
                              voting='soft',
                              weights=[5,1,3,1,5,2])
accuracy_test_matrix=[]
for i in range(10):
    accuracy_matrix_line1 = []
    accuracy_matrix_line2 = []
    xa = feature_list_total[i]
    y = label_list_total[i]
    print('Training model using subject S{} data'.format(i+1))
    print(xa.shape,y.shape)
    X_train_a, X_test_a, Y_train, Y_test = train_test_split(xa, y, test_size=0.4, random_state=random_seed)
    voting_clf.fit(X_train_a, Y_train)
    y_pred = voting_clf.predict(X_train_a)
    print('Accuracy Voting Classifier in train score',balanced_accuracy_score(Y_train, y_pred))
    accuracy_matrix_line2.append(balanced_accuracy_score(Y_train, y_pred))
    y_pred = voting_clf.predict(X_test_a)
    print('Accuracy Voting Classifier in test score',balanced_accuracy_score(Y_test, y_pred))
    accuracy_matrix_line1.append(balanced_accuracy_score(Y_test, y_pred))
    # print(sum(y_pred),sum(Y_train)/len(Y_train))
    for clf_name, clf in voting_clf.named_estimators_.items():
        train_pred = clf.predict(X_train_a)
        test_pred = clf.predict(X_test_a)
        train_score = balanced_accuracy_score(Y_train, train_pred)
        test_score = balanced_accuracy_score(Y_test, test_pred)
        print(f"{clf_name} train balanced accuracy: {train_score}")
        print(f"{clf_name} test balanced accuracy: {test_score}")
        accuracy_matrix_line1.append(test_score)
        accuracy_matrix_line2.append(train_score)

    accuracy_matrix_line=accuracy_matrix_line1+accuracy_matrix_line2
    accuracy_test_matrix.append(accuracy_matrix_line)

    # joblib.dump(filename='voting_classifier{}.model'.format(NUM),value=voting_clf)

df = pd.DataFrame(accuracy_test_matrix)

df.to_csv('classifier_acc1.csv', index=False, header=False)
