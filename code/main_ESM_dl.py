# -*- coding: utf-8 -*-
import pandas as pd
import torch
import seaborn as sns
import warnings
# warnings.filterwarnings("ignore")
import numpy as np    
import random
import os
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
import esm
random.seed(1)
from models_2 import get_model_dna_pro_att_torch
from metrics import scores
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt

model_1 = False
EPOCHS=5000
AUG = True
INIT_LR=1e-5
BATCH_SIZE = 32
criterion = BCEWithLogitsLoss()
best_model_path = 'best_model.pth'  # 最佳模型保存路径

# 创建保存模型的目录
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
# 全局变量收集真实标签和预测分数
all_y_true = []
all_y_scores = []
from torch.optim import Adam
# def get_model_dna_pro_att_torch(INIT_LR, EPOCHS, shape0, shape1, shape2, shape3, shape4,shape5,shape6):
#     model = ProteinModel(shape0, shape1, shape2, shape3, shape4, shape5, shape6)
#     optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
#     return model, optimizer

def load_validation_data(validation_path):
    """加载独立验证集数据"""
    data_val = []
    with open(validation_path, "r") as f:
        for i in f.readlines():
            tem = i.split(",")
            tem[1] = tem[1].split("\n")[0]
            tem = tuple(tem)
            data_val.append(tem)
    data_val_2 = []
    for i in data_val:
        tem = list(i)
        tem.append(1 if 'posi' in validation_path else 0)
        data_val_2.append(tem)

    return data_val, data_val_2


def extract_features(data, model, alphabet, device):
    """使用ESM模型提取特征"""
    batch_converter = alphabet.get_batch_converter()
    sequences = data
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_embeddings = results["representations"][33]
    per_residue_embeddings = token_embeddings[:, 1:-1, :]
    sequence_embedding = per_residue_embeddings.mean(dim=1)

    return sequence_embedding

def plot_training_history(train_loss_history, val_accuracy_history):
    plt.figure(figsize=(12, 5))
    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracy_history, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def reshapes(X_en_tra,X_pr_tra):
    sq=int(math.sqrt(X_en_tra.shape[1]))
    if pow(sq,2)==X_en_tra.shape[1]:
        X_en_tra2=X_en_tra.reshape((-1,sq,sq))
        X_pr_tra2=X_pr_tra.reshape((-1,sq,sq))
    else:
        X_en_tra2=np.concatenate((X_en_tra,np.zeros((X_en_tra.shape[0],int(pow(sq+1,2)-X_en_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
        X_pr_tra2=np.concatenate((X_pr_tra,np.zeros((X_pr_tra.shape[0],int(pow(sq+1,2)-X_pr_tra.shape[1])))),
                                  axis=1).reshape((-1,sq+1,sq+1))
    return X_en_tra2, X_pr_tra2

def evaluate_on_validation_set(model, validation_data, validation_labels, device):
    """在独立验证集上评估模型"""
    model.eval()
    val_dataset = CustomDataset(validation_data, validation_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_scores = []

    with torch.no_grad():
        for (esm_batch), y_batch in val_loader:
            esm_batch = esm_batch.to(device)
            outputs = model(esm_batch).flatten()
            y_scores.extend(torch.sigmoid(outputs).cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())

    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Validation Set')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

####    TSNE
def tsne_plot(data,data2):
    # data = all_data_pd
    data = pd.DataFrame(data)
    X, y = data.iloc[:, :].to_numpy(), np.array([i[2] for i in data2])
    print(X[0])
    print(y[0:10])
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
    plt.title('Original')
    # plt.title('3x augmentation')
    plt.colorbar(scatter)
    plt.show()


def plot_tsne(features, labels, title="t-SNE Visualization"):
    """
    绘制t-SNE图
    features: 特征矩阵 [n_samples, n_features]
    labels: 样本标签 [n_samples]
    """
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # 绘制
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.7,
        s=50
    )
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Class")
    plt.grid(True)
    plt.show()
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.esm_data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.esm_data[idx]), self.labels[idx]


def plot_combined_tsne(train_features, train_labels, val_features, val_labels, title="Combined t-SNE"):
    """
    绘制训练集和验证集的 t-SNE 结果（同一张图）
    """
    # 合并特征和标签
    combined_features = np.concatenate([train_features, val_features], axis=0)
    combined_labels = np.concatenate([train_labels, val_labels], axis=0)

    # 标记训练集和验证集（训练集=0，验证集=1）
    dataset_type = np.concatenate([
        np.zeros(len(train_features)),  # 训练集标记为0
        np.ones(len(val_features))  # 验证集标记为1
    ])

    # 执行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_features)

    # 绘制散点图
    plt.figure(figsize=(10, 8))

    # 绘制训练集（红色）
    plt.scatter(
        tsne_results[dataset_type == 0, 0],  # 训练集数据
        tsne_results[dataset_type == 0, 1],
        c=train_labels,  # 颜色表示类别
        cmap='viridis',
        marker='o',  # 训练集用圆圈
        alpha=0.6,
        label='Training Set'
    )

    # 绘制验证集（蓝色，用不同标记）
    plt.scatter(
        tsne_results[dataset_type == 1, 0],  # 验证集数据
        tsne_results[dataset_type == 1, 1],
        c=val_labels,  # 颜色表示类别
        cmap='viridis',
        marker='s',  # 验证集用方块
        alpha=0.6,
        label='Validation Set'
    )

    plt.colorbar(label='Class Label')
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()

def newmodel_dna_and_pro_and_att(X_tra_pro, y_tra3, X_val_pro, y_val3, shape6, device):
    model, optimizer = get_model_dna_pro_att_torch(INIT_LR, EPOCHS, shape6)
    model = model.to(device)
    # 存储倒数第二层特征
    penultimate_features_train = []
    penultimate_features_val = []

    best_val_accuracy = 0.0
    best_epoch = 0

    train_dataset = CustomDataset(X_tra_pro, y_tra3)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(X_val_pro, y_val3)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_loss_history = []
    val_accuracy_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for (esm_batch), y_batch in train_loader:
            esm_batch, y_batch = esm_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output, penultimate = model(esm_batch)
            penultimate_features_train.append(penultimate.detach().cpu().numpy())  # <-- 新增
            loss = criterion(output.squeeze(-1), y_batch.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * esm_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # 验证阶段
        model.eval()
        y_pred_val = []
        y_true_val = []
        y_score_val = []

        with torch.no_grad():
            for (esm_batch), y_batch in val_loader:
                esm_batch, y_batch = esm_batch.to(device), y_batch.to(device)
                # outputs = model(esm_batch).flatten()
                outputs, penultimate = model(esm_batch)

                y_score_val.extend(torch.sigmoid(outputs).cpu().numpy())
                y_pred_val.extend(outputs.cpu().numpy() > 0.5)
                y_true_val.extend(y_batch.cpu().numpy())
                # 存储验证集倒数第二层特征
                penultimate_features_val.append(penultimate.detach().cpu().numpy())

        y_pred_val = np.array(y_pred_val) > 0.5
        val_accuracy = np.mean(y_pred_val == y_true_val)
        val_accuracy_history.append(val_accuracy)
        # 合并所有batch的特征
        penultimate_train = np.concatenate(penultimate_features_train, axis=0)
        penultimate_val = np.concatenate(penultimate_features_val, axis=0)
        # 绘制训练集t-SNE
        plot_tsne(penultimate_train, y_tra3, title="Training Set Penultimate Layer t-SNE")

        # 绘制验证集t-SNE
        plot_tsne(penultimate_val, y_val3, title="Validation Set Penultimate Layer t-SNE")
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_accuracy': val_accuracy
            }, best_model_path)

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}")

    print(f"Best model saved at epoch {best_epoch + 1} with validation accuracy {best_val_accuracy:.4f}")
    return train_loss_history, val_accuracy_history


def main():
    # 初始化设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    # 加载数据
    data_posi = []
    data_nega = []
    with open("../data/posi_9.csv", "r") as f:
        for i in f.readlines():
            tem = i.split(",")
            tem[1] = tem[1].split("\n")[0]
            tem = tuple(tem)
            data_posi.append(tem)

    data_posi_2 = []
    for i in data_posi:
        tem = list(i)
        tem.append(1)
        data_posi_2.append(tem)

    with open("../data/nega_9.csv", "r") as f:
        for i in f.readlines():
            tem2 = i.split(",")
            tem2[1] = tem2[1].split("\n")[0]
            tem2 = tuple(tem2)
            data_nega.append(tem2)

    data_nega_2 = []
    for i in data_nega:
        tem = list(i)
        tem.append(0)
        data_nega_2.append(tem)

    data_test = data_posi[1:] + data_nega[1:]
    data_test_2 = data_posi_2[1:] + data_nega_2[1:]

    # 提取特征
    sequence_embedding = extract_features(data_test, model, alphabet, device)

    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for train_index, test_index in kf.split(data_test):
        X_tra = np.array([sequence_embedding[ii] for ii in train_index])
        X_val = np.array([sequence_embedding[ii] for ii in test_index])
        y_tra = np.array([data_test_2[ii][2] for ii in train_index])
        y_val = np.array([data_test_2[ii][2] for ii in test_index])

        X_tra_esm, X_val_esm = reshapes(X_tra, X_val)
        X_esm = X_tra_esm.reshape(X_tra_esm.shape[0], X_tra_esm.shape[1], X_tra_esm.shape[2], 1)

        alldata_aug = [(X_esm[i, :, :, :], y_tra[i]) for i in range(len(X_tra_esm))]
        random.shuffle(alldata_aug)

        DNA_allfeatures, labels_aug = np.array([i[0] for i in alldata_aug]), [i[1] for i in alldata_aug]
        X_val = X_val_esm.reshape(X_val_esm.shape[0], X_val_esm.shape[1], X_val_esm.shape[2], 1)

        # 训练模型
        train_loss, val_accuracy = newmodel_dna_and_pro_and_att(
            DNA_allfeatures, labels_aug, X_val, y_val, 1, device
        )

        # 绘制训练曲线
        plot_training_history(train_loss, val_accuracy)
    # 加载最佳模型并在独立验证集上测试
    validation_path = "../data/validation.csv"  # 替换为你的验证集路径
    validation_data, validation_labels = load_validation_data(validation_path)

    # 提取验证集特征
    val_sequence_embedding = extract_features(validation_data, model, alphabet, device)
    X_val_esm = val_sequence_embedding.reshape(val_sequence_embedding.shape[0], -1)
    X_val_esm = X_val_esm.reshape(X_val_esm.shape[0], int(np.sqrt(X_val_esm.shape[1])),
                                  int(np.sqrt(X_val_esm.shape[1])))
    X_val_esm = X_val_esm.reshape(X_val_esm.shape[0], X_val_esm.shape[1], X_val_esm.shape[2], 1)
    y_val = np.array([i[2] for i in validation_labels])

    # 加载最佳模型
    checkpoint = torch.load(best_model_path)
    best_model, _ = get_model_dna_pro_att_torch(INIT_LR, EPOCHS, 1)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)

    # 在验证集上评估
    roc_auc = evaluate_on_validation_set(best_model, X_val_esm, y_val, device)
    print(f"Validation ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
# def newmodel_dna_and_pro_and_att(X_tra_pro, y_tra3, X_val_pro,  y_val3, shape6):
#     model = None
#     model, optimizer = get_model_dna_pro_att_torch(INIT_LR, EPOCHS, shape6)
#     # model.summary()
#     # print(model)
#     print('Traing model ...')
#     ##### 自定义数据集 #####
#     train_loss_history = []
#     val_accuracy_history = []
#     class CustomDataset(Dataset):
#         def __init__(self, data, labels):
#             self.esm_data = torch.tensor(data, dtype=torch.float32)
#             # self.ct_data = torch.tensor(ct_data, dtype=torch.float32)
#             # self.cks_data = torch.tensor(cks_data, dtype=torch.float32)
#             self.labels = torch.tensor(labels, dtype=torch.float32)
#
#         def __len__(self):
#             return len(self.labels)
#
#         def __getitem__(self, idx):
#             return (self.esm_data[idx]), self.labels[idx]
#
#     ##### 创建数据加载器 #####
#     ##### training #####
#     train_dataset = CustomDataset(X_tra_pro, y_tra3)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     ##### validation #####
#     val_dataset = CustomDataset( X_val_pro, y_val3)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     for epoch in range(EPOCHS):
#         model.train()
#         epoch_loss = 0.0
#         for (esm_batch), y_batch in train_loader:
#             # dde_batch,ct_batch,cks_batch,y_batch = dde_batch.to(device), ct_batch,cks_batch.cuda()
#             optimizer.zero_grad()
#             output = model(esm_batch).flatten()
#             loss = criterion(output.squeeze(-1), y_batch.float())
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item() * esm_batch.size(0)
#             # 计算平均 epoch loss
#             epoch_loss /= len(train_loader.dataset)
#             train_loss_history.append(epoch_loss)
#         model.eval()
#         y_pred_val = []
#         y_true_val = []
#         y_score_val = []
#         with torch.no_grad():
#             for (esm_batch), y_batch in val_loader:
#                 # dna_batch, pro_batch = dna_batch.to(device), pro_batch.to(device)
#                 outputs = model(esm_batch).flatten()
#                 y_score_val.extend(torch.sigmoid(outputs).cpu().numpy())
#                 y_pred_val.extend(outputs.cpu().numpy()>0.5)
#                 y_true_val.extend(y_batch.cpu().numpy())
#                 # 计算准确度（假设阈值为 0.5）
#         # 收集每个fold的预测结果
#         if epoch == EPOCHS - 1:  # 只在最后一个epoch收集
#             all_y_true.extend(y_true_val)
#             all_y_scores.extend(y_score_val)
#
#         y_pred_val = np.array(y_pred_val) > 0.5
#         val_accuracy = np.mean(y_pred_val == y_true_val)
#         val_accuracy_history.append(val_accuracy)
#     # y_pred_val = np.array(y_pred_val)  # 转换为 NumPy 数组，与 Keras 输出一致
#     # train_model(model,[X_tra_dna,X_tra_pro,X_tra_ct], y_tra3)
#     # y_pred_val = predict(model,[X_val_dna,X_val_pro,X_val_ct])
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#     return train_loss_history, val_accuracy_history
#
#
# import matplotlib.pyplot as plt
#
#

#
# def obtainfeatures(data,file_path1,file_path2,strs):
#     phage_features=[]
#     host_features=[]
#     labels=[]
#     for i in data:
#         phage_features.append(np.loadtxt(file_path1+i[0]+strs).tolist())
#         host_features.append(np.loadtxt(file_path2+i[1].split('.')[0]+strs).tolist())
#         labels.append(i[-1])
#     return np.array(phage_features), np.array(host_features), np.array(labels)
#
# def obtain_neg(X_tra,X_val):
#     X_tra_pos=[mm for mm in X_tra if mm[2]==1]
#     X_neg=[str(mm[0])+','+str(mm[1]) for mm in X_tra+X_val if mm[2]==0]
#     training_neg=[]
#     phage=list(set([mm[0]for mm in X_tra_pos]))
#     host=list(set([mm[1]for mm in X_tra_pos]))
#     for p in phage:
#         for h in host:
#             if str(p)+','+str(h) in X_neg:
#                 continue
#             else:
#                 training_neg.append([p,h,0])
#     return random.sample(training_neg,len(X_tra_pos))
#
# result_all=[]
# pred_all=[]
# test_y_all=[]
#
# # from feature_extraction import DDE, CKSAAP, CTriad, readFasta
# ###################
# ##### 数据读取 #####
# ##################
# # data_posittive = pd.read_csv("./data_augment/posi_8.csv")
# # data_negative = pd.read_csv("../data/sig_nega.csv")
# # print(len(data_negative))
# #
# # DDE_feature_neg = DDE(data_negative)
# # CKSAAP_feature_neg = CKSAAP(data_negative)
# # CTriad_feature_neg = CTriad(data_negative)
# #
# # ####################
# # ##### 数据处理 ######
# # ###################
# # DDE_feature_neg = DDE_feature_neg[1:]
# # CKSAAP_feature_neg = CKSAAP_feature_neg[1:]
# # CTriad_feature_neg = CTriad_feature_neg[1:]
# #
# # data_list_dde = [i[2:] for i in DDE_feature_neg]
# # data_list_cks = [i[2:] for i in CKSAAP_feature_neg]
# # data_list_ct = [i[2:] for i in CTriad_feature_neg]
# # if model_1 == True:
# #     data_all = [data_list_dde[i] + data_list_cks[i] + data_list_ct[i]  for i in range(len(data_list_dde))]
# #     data_all = np.array(data_all)
# #     print(len(data_all), len(data_all[0]))
# # data_list_dde = np.array(data_list_dde)
# # data_list_cks = np.array(data_list_cks)
# # data_list_ct = np.array(data_list_ct)
# # #####################
# # ##### 增强数据集 #####
# # ####################
# # data=pd.read_csv('../result/result_GAN/Iteration_50000.txt',header=None,sep=',')
# # data_dde = data.iloc[:,0:400]   ### (220,400)
# # data_cks = data.iloc[:,400:1600] ### (220,1200)
# # data_ct = data.iloc[:,1600:1943] ### (220,343)
# #
# # data1=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',')  #### 624
# # data1=data1[data1[2]==1]
# # allinter=[str(data1.loc[i,0])+','+str(data1.loc[i,1]) for i in data1.index] ### 312 list
# #
# # ### (312,1004)
# # newdata=pd.read_csv('../result/result_GAN/Iteration_99800.txt',sep=',',header=None).values[:,:-1].tolist()   ##optimal pseudo samples
# # dic_newdata={}
# # for i in range(len(newdata)):
# #     dic_newdata[allinter[i]]=newdata[i]
# #     ### {1:[1004]} 312
# data_posi = []
# # data_test_2 = []
# data_nega = []
# # data_nega_2 = []
# with open("../data/posi_8.csv", "r") as f:
#     for i in f.readlines():
#         tem = i.split(",")
#         tem[1] = tem[1].split("\n")[0]
#         # tem.append(1)
#         tem = tuple(tem)
#         data_posi.append(tem)
# # data_test = data_test[1:]
# data_posi_2 = []
# for i in data_posi:
#     tem = list(i)
#     tem.append(1)
#     data_posi_2.append(tem)
# # print(data_test)  # Check if the file opens correctly
# with open("../data/nega_8.csv", "r") as f:
#     for i in f.readlines():
#         tem2 = i.split(",")
#         tem2[1] = tem2[1].split("\n")[0]
#         # tem2.append(0)
#         tem2 = tuple(tem2)
#         data_nega.append(tem2)
# # data_nega_2 = [list(i).append(0) for i in data_nega]
# data_nega_2 = []
# print("negative数据集规模：", len(data_posi))
# for i in data_nega:
#     tem = list(i)
#     tem.append(0)
#     data_nega_2.append(tem)
#
# data_test = data_posi[1:] + data_nega[1:]
# data_test_2 = data_posi_2[1:] + data_nega_2[1:]
# print(len(data_test_2))
# # print(data_test)  # Check if the file opens correctly
# print("123")
# current_date_and_time = datetime.now()
# print("embeding之前：", current_date_and_time)
# # 加载模型和 tokenizer
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# batch_converter = alphabet.get_batch_converter()
# model.eval()  # 切换到推理模式
#
# # 检查是否使用 GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# sequences = data_test;
# batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
# batch_tokens = batch_tokens.to(device)
#
# with torch.no_grad():
#     results = model(batch_tokens, repr_layers=[33])
# token_embeddings = results["representations"][33]  # (batch_size, seq_len+2, 1280)
#
# # 获取每个氨基酸的特征（去掉 CLS 和 EOS 标记）
# per_residue_embeddings = token_embeddings[:, 1:-1, :]  # (1, seq_len, 1280)
# # print(per_residue_embeddings.shape)
# # 获取整个序列的全局特征（平均池化）
# sequence_embedding = per_residue_embeddings.mean(dim=1)  # (1, 1280)
# print(sequence_embedding.shape)
#
# current_date_and_time = datetime.now()
# print("embeding之后：", current_date_and_time)
#
# kf = KFold(n_splits=5,shuffle=True, random_state=1)
# # training=pd.read_csv('../data/data_pos_neg.txt',header=None,sep=',').values.tolist() ### 624 [[],[],...]
# # data_posittive, data_negative = pd.read_csv("../data/posi_8.csv").values.tolist(), pd.read_csv("../data/sig_nega.csv")[:1800].values.tolist()
# # data_posittive = [i+[1] for i in data_posittive]
# # print(len(data_posittive),data_posittive[0])
# # data_negative = [i+[0] for i in data_negative]
# # training = data_posittive + data_negative
# for train_index, test_index in kf.split(data_test):  ### list [index] 4:1
#     ###obtain data
#
#     X_tra=np.array([sequence_embedding[ii] for ii in train_index])   ### [[3],[3],[3],...] 499    ###zzy.
#     X_val=np.array([sequence_embedding[ii] for ii in test_index])   ### [[3],[3],[3],...] 125
#     # neg_select=obtain_neg(X_tra,X_val)  ##add extra negative samples   [[3],[3],...]  258
#     y_tra = np.array([data_test_2[ii][2] for ii in train_index])
#
#     y_val = np.array([data_test_2[ii][2] for ii in test_index])
#
#     X_tra_esm, X_val_esm = reshapes(X_tra, X_val)
#     print(X_tra_esm.shape) ### [3600, 1280]
#     print(X_val_esm.shape)
#     X_esm = X_tra_esm.reshape(X_tra_esm.shape[0], X_tra_esm.shape[1], X_tra_esm.shape[2], 1)
#     # print(X_esm.shape)
#     # X_dde=np.array([X_tra_dde2,X_val_dde2]).transpose(1,2,3,0)
#     # y_tra = np.concatenate((np.ones((len(X_tra_dde),)), np.ones((len(X_tra_dde),))), axis=0)
#     # y_tra_aug = np.concatenate((np.ones((len(select_tra),)), np.ones((len(select_tra),))), axis=0)
#     ### (757,13,13,2)
#     alldata_aug=[(X_esm[i,:,:,:],y_tra[i]) for i in range(len(X_tra_esm))]
#     print(len(alldata_aug))
#     random.shuffle(alldata_aug)
#     ### a => (1015, 19, 19, 2); b => (1015, 13, 13, 2); c => 1015
#
#     DNA_allfeatures,labels_aug=np.array([i[0] for i in alldata_aug]), [i[1] for i in alldata_aug]
#     ### 125 labels
#     test_y_all=test_y_all+y_val.tolist()
#     X_val = X_val_esm.reshape(X_val_esm.shape[0], X_val_esm.shape[1], X_val_esm.shape[2], 1)
#     print(len(X_val))
#     # X_val_cks3 = X_val_cks2.reshape(X_val_cks2.shape[0], X_val_cks2.shape[1], X_val_cks2.shape[2], 1)
#     # X_val_ct3 = X_val_ct2.reshape(X_val_ct2.shape[0], X_val_ct2.shape[1], X_val_ct2.shape[2], 1)
#     ##prediction model
#     phiaf_result,phiaf_pred=newmodel_dna_and_pro_and_att(DNA_allfeatures, labels_aug, X_val, y_val,1)
#     plot_training_history(phiaf_result, phiaf_pred)
#     result_all.append(phiaf_result)
#     # print("结果1：",result_all)
#     print("结果2：", len(result_all[0]))
#     pred_all=pred_all+phiaf_pred
#     # print("结果3：",pred_all)
#     print("结果4：", len(pred_all))
#
# # 在kfold循环结束后添加ROC绘制代码
# if len(all_y_true) > 0 and len(all_y_scores) > 0:
#     fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
#     roc_auc = auc(fpr, tpr)
#
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")
#     plt.show()
# plot_training_history(train_loss, val_accuracy)


