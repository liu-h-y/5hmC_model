
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np


if __name__ == '__main__':
    df_train = pd.read_csv("weakRM_data/feature/train.csv", header=None)
    X_train = df_train.iloc[:,1:]
    y_train = df_train.iloc[:,0]
    X_train = X_train.values
    y_train = y_train.values

    df_valid = pd.read_csv("weakRM_data/feature/valid.csv", header=None)
    X_valid = df_valid.iloc[:, 1:]
    y_valid = df_valid.iloc[:, 0]
    X_valid = X_valid.values
    y_valid = y_valid.values

    c_list = np.logspace(-5, 15, 21, base=2)
    c_list = c_list.tolist()

    g_list = np.logspace(-15, -5, 11, base=2)
    g_list = g_list.tolist()

    sum_tasks = len(c_list)*len(g_list)
    count = 0

    fo = open("res/res_c_g.txt", 'w')
    fo.write("c\tgamma\tACC\tSn\tSp\tAUC\tAP\tMCC\tTP\tTN\tFP\tFN\n")

    for c in c_list:
        for g in g_list:
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_valid)

            # print c,g
            clf = SVC(C=c, gamma=g, probability=True)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_valid)
            y_predict_prob = clf.predict_proba(X_valid)
            y_predict_prob = y_predict_prob[:, 1]

            TN, FP, FN, TP = (metrics.confusion_matrix(y_valid, y_predict)).ravel()
            TN = float(TN)
            FP = float(FP)
            FN = float(FN)
            TP = float(TP)

            final_test_accuracy = (TP + TN) / (TN + FP + FN + TP)
            final_test_sensitivity = TP / (TP + FN)
            final_test_specificity = TN / (FP + TN)
            final_test_auc = metrics.roc_auc_score(y_valid, y_predict_prob)
            final_test_ap = metrics.average_precision_score(y_valid, y_predict_prob)
            final_test_mcc = metrics.matthews_corrcoef(y_valid, y_predict)

            fo.write(str(c) + "\t" + str(g) + "\t" + str(final_test_accuracy) \
                     + "\t" + str(final_test_sensitivity) + "\t" + str(final_test_specificity) + "\t" + str(final_test_auc) \
                     + "\t" +str(final_test_ap)+ "\t" + str(final_test_mcc) + "\t" + str(TP) + "\t" + str(TN)+ "\t" + str(FP)+ "\t" + str(FN)+"\n")

            count += 1
            print("progress :"+str(count)+"of"+str(sum_tasks))

    fo.close()


