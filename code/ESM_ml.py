# -*- coding: utf-8 -*-
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime
import esm
from plot_roc import plot_roc_cv
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier as XGB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
random.seed(1)
import math
data_posi = []
# data_test_2 = []
data_nega = []
# data_nega_2 = []

ESM_data = False

with open("../data/posi_0.csv", "r") as f:
    for i in f.readlines():
        tem = i.split(",")
        tem[1] = tem[1].split("\n")[0]
        # tem.append(1)
        tem = tuple(tem)
        data_posi.append(tem)
# data_test = data_test[1:]
data_posi_2 = []
for i in data_posi:
    tem = list(i)
    tem.append(1)
    data_posi_2.append(tem)
# print(data_test)  # Check if the file opens correctly
with open("../data/nega_1.csv", "r") as f:
    for i in f.readlines():
        tem2 = i.split(",")
        tem2[1] = tem2[1].split("\n")[0]
        # tem2.append(0)
        tem2 = tuple(tem2)
        data_nega.append(tem2)
        # data_nega = data_nega[:201]
print("negative:", len(data_nega))
# print(data_nega[0])
        # data_nega = data_nega[:200]
# data_nega_2 = [list(i).append(0) for i in data_nega]
data_nega_2 = []
print("positive数据集规模：", len(data_posi))
for i in data_nega:
    tem = list(i)
    tem.append(0)
    data_nega_2.append(tem)

data_test = data_posi[1:] + data_nega[1:]
print(data_test[:4], len(data_test))
# print("test:", data_test[0])
# data_test_2 = data_posi_2[1:] + data_nega_2[1:]
# print(len(data_test_2))
# print(data_test)  # Check if the file opens correctly

###########################################
######## validation datasets ##############
###########################################
validation_list = []
validation_data = pd.read_csv("../data/validation.csv")
print(validation_data)
for i in range(len(validation_data["id"])):
    validation_list.append((validation_data["id"][i],validation_data["seq"][i]));

print("Begin:123")
if ESM_data:
    with open("../result/result_GAN/Iteration_18000.txt") as f:
        MatrixFeaturesPositive = [list(x.split(" ")) for x in f]
    FeaturesPositive = [line[:] for line in MatrixFeaturesPositive[:]]
    # print(FeaturesPositive[199][0].split(",")[1280])
    feature_positive = [];
    for i in FeaturesPositive:
        tem = i[0].split(",")[:1280]
        tem = [float(i) for i in tem]
        # print(len(tem))
        feature_positive.append(tem)
    print(len(feature_positive[0]))
    dataset_augment = np.array(feature_positive, dtype='float32')
    print(dataset_augment.shape)

current_date_and_time = datetime.now()
print("embeding之前：", current_date_and_time)
# 加载模型和 tokenizer
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # 切换到推理模式
# 检查是否使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# sequences = data_test;
############################
#########自定义函数##########
###########################
def esm_coding(sequences):
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_embeddings = results["representations"][33]  # (batch_size, seq_len+2, 1280)

    # 获取每个氨基酸的特征（去掉 CLS 和 EOS 标记）
    per_residue_embeddings = token_embeddings[:, 1:-1, :]  # (1, seq_len, 1280)
    # print(per_residue_embeddings.shape)
    # 获取整个序列的全局特征（平均池化）
    sequence_embedding = per_residue_embeddings.mean(dim=1)  # (1, 1280)
    # print(sequence_embedding[0])
    # print(sequence_embedding.shape)
    current_date_and_time = datetime.now()
    print("embeding之后：", current_date_and_time)
    if ESM_data:
        all_data = np.concatenate([dataset_augment, sequence_embedding], axis=0)  # 沿第0轴(行方向)拼接
        all_data = all_data.tolist()
    else:
        all_data = sequence_embedding.tolist()
    new_all_data = []
    # print("test:", len(all_data[0]))
    # print("train:", all_data[0])
    for i in range(len(all_data)):
        if i < (len(all_data) / 2):
            new_all_data.append(all_data[i].append(1))
        else:
            new_all_data.append(all_data[i].append(0))
    all_data_pd = pd.DataFrame(all_data)
    return all_data_pd
# print("padas dataframe:", all_data_pd.shape)
all_data_pd = esm_coding(data_test)
all_data_pd_in = esm_coding(validation_list)
########################################
############### 自定义函数 ###############
########################################

def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'Sensitivity': 'NA',
        'Specificity': 'NA',
        'Accuracy': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['Sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else 'NA'
    my_metrics['Specificity'] = tn / (fp + tn) if (fp + tn) != 0 else 'NA'
    my_metrics['Accuracy'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (
        tp + fn) * (tn + fp) * (tn + fn) != 0 else 'NA'
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 'NA'
    my_metrics['Recall'] = my_metrics['Sensitivity']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 'NA'
    # print(my_metrics)
    return my_metrics
####    svm
def svm_self(data_pd, types, b=1, n=0):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, :-1], data_pd.iloc[n:,-1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    roc_auc = []
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # clf = XGB(n_estimators=100)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC(kernel='rbf', degree=3)
        # clf = LogisticRegression()
        clf = svm.SVC(probability=True)
        clf.fit(X_train, y_train)
        # joblib.dump(clf, "./svm_model/" + types + ".pkl")
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
        roc_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum1 = 0
        for j in range(len(metrics)):
            if (metrics[j][i] == 'NA'):
                sum1 += 0
            else:
                sum1 += metrics[j][i]
        my_metrics.append(sum1/10)
    print(my_metrics)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y,sum(roc_auc) / len(roc_auc)
####    RF
def ml(data_pd, types, b=1, n=0):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, :-1], data_pd.iloc[n:,-1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    roc_auc = []
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=100
                                     ,random_state=0,criterion='gini',min_samples_leaf=4)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
        roc_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        acc.append(accuracy_score(y_test, clf.predict(X_test)))

    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum1 = 0
        for j in range(len(metrics)):
            sum1 += metrics[j][i]
        my_metrics.append(sum1/10)
    print(my_metrics)
    # print(roc_auc)
    # 打印特征得分
    feature_importances = clf.feature_importances_
    # for feature_index, importance in enumerate(feature_importances):
    #     print(f"Feature {feature_index} score: {importance}")
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test), y, sum(roc_auc) / len(roc_auc)
###     MLP
def mlp(data_pd, types, b=1, n=0):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, :-1], data_pd.iloc[n:,-1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    roc_auc = []
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = MLPClassifier()
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
        roc_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        acc.append(accuracy_score(y_test, clf.predict(X_test)))

    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum1 = 0
        for j in range(len(metrics)):
            sum1 += metrics[j][i]
        my_metrics.append(sum1/10)
    print(my_metrics)
    # print(roc_auc)
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test), y, sum(roc_auc) / len(roc_auc)
###     xgboost
def xgboost(data_pd, types, data_in, b=1, n=0):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, :-1], data_pd.iloc[n:,-1].astype("int")
    # if data_in != 0:
    X_in, y_in = data_in.iloc[n:, :-1], data_in.iloc[n:, -1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    # print(X.shape)
    X_indepedent, y_indepedent = np.array(X_in), np.array(y_in)
    X_all = np.vstack([X, X_indepedent])
    y_all = np.concatenate([y,y_indepedent])
    cv = StratifiedKFold(n_splits=10)
    # parameter_space = {
    #     "n_estimators":[10,15,20],
    #     "criterion": ["gini","entropy"],
    #     "min_samples_leaf": [2,4,6]
    # }
    roc_auc = []
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = XGB(n_estimators=100)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC()
        clf.fit(X_train, y_train)
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
        roc_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
    # a = clf.predict_proba(X_indepedent)[:, 1]
    # plot_roc.plot_roc_cv(y_indepedent, a, "test","./test.svg")

    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []
    for i in metrics[0].keys():
        sum1 = 0
        for j in range(len(metrics)):
            sum1 += metrics[j][i]
        my_metrics.append(sum1/10)
    # print(my_metrics)
    # joblib.dump(clf, types + '.pkl');
    # h = clf.predict_proba(X)
    print("jieguo:",calculate_metrics(y_indepedent.tolist(), clf.predict(X_indepedent)))
    a = clf.predict_proba(X_indepedent)[:, 1]
    plot_roc_cv(y_indepedent, a, "test", "./test.svg")
    print("roc", roc_auc_score(y_indepedent, clf.predict_proba(X_indepedent)[:, 1]))
    print("test:",my_metrics)
    # print(X+X_indepedent)
    b = clf.predict_proba(X_all)[:, 1]
    plot_roc_cv(y_all, b, "test_2", "./test_2.svg")
    return clf.predict(X), importance, X_test, y_test, clf.predict(X_test),y,sum(roc_auc) / len(roc_auc)
####    LR
def LR_demo(data_pd, types, b=1, n=0):
    acc = []
    metrics = []
    X,y = data_pd.iloc[n:, :-1], data_pd.iloc[n:, -1].astype("int")
    # X_indepedent, y_indepedent = data_in.iloc[n:, 2:], data_in.iloc[n:, 1].astype("int")
    # print(X)
    X, y = np.array(X),np.array(y)
    # X_indepedent, y_indepedent = np.array(X_indepedent),np.array(y_indepedent)
    # print(y_indepedent)
    cv = StratifiedKFold(n_splits=10)
    roc_auc = []
    for train_index, test_index in cv.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # clf = XGB(n_estimators=100)
        # grid = GridSearchCV(clf,parameter_space)
        # clf = svm.SVC(kernel='rbf', degree=3)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        # print(y_test)
        # joblib.dump(clf, "./svm_model/" + types + ".pkl")
        metrics.append(calculate_metrics(y_test.tolist(),clf.predict(X_test)))
        roc_auc.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        acc.append(accuracy_score(y_train, clf.predict(X_train)))
    # a = clf.predict_proba(X_indepedent)[:, 1]
    # plot_roc.plot_roc_cv(y_indepedent, a, "test","./test.svg")
    importance = 0;
    if b == 0:
        importance = clf.feature_importances_;
    # print(acc)
    my_metrics = []

    for i in metrics[0].keys():
        sum1 = 0
        for j in range(len(metrics)):
            if (metrics[j][i] == 'NA'):
                sum1 += 0
            else:
                sum1 += metrics[j][i]
        my_metrics.append(sum1/10)
    # print(calculate_metrics(y_indepedent.tolist(), clf.predict(X_indepedent)))
    a = clf.predict_proba(X)[:, 1]
    # plot_roc.plot_roc_cv(y, a, "test", "./test.svg")
    # print(roc_auc_score(y_indepedent, clf.predict_proba(X_indepedent)[:, 1]))
    print(my_metrics)
    # return clf.predict(X_indepedent), importance, X_test, y_test, clf.predict(X_test),y,sum(roc_auc) / len(roc_auc)
####    TSNE
def tsne_plot(data):
    # data = all_data_pd
    X, y = data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy()
    print(X[0])
    print(y[0])
    # return X, y
    y_list = np.array(y).astype(int)  # 关键修正点
    tsne = TSNE(
        n_components=2,  # 输出维度
        perplexity=30,  # 建议范围5-50，控制局部结构
        n_iter=1000,  # 优化迭代次数
        random_state=42  # 设置随机种子保证可重复性
    )
    X_tsne = tsne.fit_transform(X)

    # 可视化结果
    plt.figure(figsize=(8, 6))
    markers = ['^', 'o']
    # colors = ['#FF0000', '#00FF00']
    # 遍历不同的类别
    for class_label in np.unique(y_list):
        # 筛选出当前类别的数据
        mask = y_list == class_label
        X_class = X_tsne[mask]
        y_class = y_list[mask]

        # 根据类别索引选择标记样式
        marker = markers[class_label]

        # 绘制当前类别的散点
        scatter = plt.scatter(X_class[:, 0], X_class[:, 1], cmap='RdYlBu', alpha=0.9, marker=marker, edgecolors='k')

    # plt.title('Original (ESM)')
    plt.title('1x augmentation (ESM)')
    plt.colorbar(scatter)
    plt.show()
# tsne_plot(all_data_pd)
##############################################
############# 模型训练与性能评估 ################
#############################################

# a  = svm_self(all_data_pd,"AAC")
# b  = ml(all_data_pd,"AAC")
# c  = mlp(all_data_pd,"AAC")
d  = xgboost(all_data_pd,"AAC",all_data_pd_in)
# e  = LR_demo(all_data_pd,"AAC")



