import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import time


if __name__ == '__main__':
    start_time = time.time()

    df_train = pd.read_csv("weakRM_data/feature/train.csv", header=None)
    X_train = df_train.iloc[:,1:]
    y_train = df_train.iloc[:,0]
    X_train = X_train.values
    y_train = y_train.values

    df_test = pd.read_csv("weakRM_data/feature/test.csv", header=None)
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, 0]
    X_test = X_test.values
    y_test = y_test.values

    c = 16
    g = 0.03125

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    # print c,g
    clf = SVC(C=c, gamma=g, probability=True)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    y_predict_prob = clf.predict_proba(X_test)
    y_predict_prob = y_predict_prob[:, 1]

    TN, FP, FN, TP = (metrics.confusion_matrix(y_test, y_predict)).ravel()
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)
    TP = float(TP)

    final_test_accuracy = (TP + TN) / (TN + FP + FN + TP)
    final_test_sensitivity = TP / (TP + FN)
    final_test_specificity = TN / (FP + TN)
    final_test_auc = metrics.roc_auc_score(y_test, y_predict_prob)
    final_test_ap = metrics.average_precision_score(y_test, y_predict_prob)
    final_test_mcc = metrics.matthews_corrcoef(y_test, y_predict)

    print("acc:")
    print(final_test_accuracy)
    print("sn")
    print(final_test_sensitivity)
    print("sp")
    print(final_test_specificity)
    print("auc")
    print(final_test_auc)
    print("ap")
    print(final_test_ap)
    print("mcc")
    print(final_test_mcc)





