import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("./i5hmCVec_on_iRNA5hmC_data/feature.csv", header=None)
X = df.iloc[:,1:]
y = df.iloc[:,0]
X = X.values
y = y.values

c = 1.0
g = 0.03125


acc = []
sn = []
sp = []
auc = []
ap = []
mcc = []

for random_num in range(10):
    sum_y_test = np.array([])
    sum_y_pre = np.array([])
    sum_y_pre_prob = np.array([])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_num)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
        sum_y_test = np.append(sum_y_test, y_test)
        sum_y_pre = np.append(sum_y_pre, y_predict)
        sum_y_pre_prob = np.append(sum_y_pre_prob, y_predict_prob)

    TN, FP, FN, TP = (metrics.confusion_matrix(sum_y_test, sum_y_pre)).ravel()
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)
    TP = float(TP)

    final_test_accuracy = (TP + TN) / (TN + FP + FN + TP)
    final_test_sensitivity = TP / (TP + FN)
    final_test_specificity = TN / (FP + TN)
    final_test_auc = metrics.roc_auc_score(sum_y_test, sum_y_pre_prob)
    final_test_ap = metrics.average_precision_score(sum_y_test, sum_y_pre_prob)
    final_test_mcc = metrics.matthews_corrcoef(sum_y_test, sum_y_pre)

    acc.append(final_test_accuracy)
    sn.append(final_test_sensitivity)
    sp.append(final_test_specificity)
    auc.append(final_test_auc)
    ap.append(final_test_ap)
    mcc.append(final_test_mcc)

acc = np.array(acc)
sn = np.array(sn)
sp = np.array(sp)
auc = np.array(auc)
ap = np.array(ap)
mcc = np.array(mcc)

print("acc mean: ",end="")
print(np.mean(acc))
print("acc std: ",end="")
print(np.std(acc))

print("sn mean: ",end="")
print(np.mean(sn))
print("sn std: ",end="")
print(np.std(sn))

print("sp mean: ",end="")
print(np.mean(sp))
print("sp std: ",end="")
print(np.std(sp))

print("auc mean: ",end="")
print(np.mean(auc))
print("auc std: ",end="")
print(np.std(auc))

print("ap mean: ",end="")
print(np.mean(ap))
print("ap std: ",end="")
print(np.std(ap))

print("mcc mean: ",end="")
print(np.mean(mcc))
print("mcc std: ",end="")
print(np.std(mcc))