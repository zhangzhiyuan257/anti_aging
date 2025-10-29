import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from code  import readFasta
from feature_extraction  import *
from other_script import *
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report,roc_curve,auc

def main():
### 1.read data
    positive_data = readFasta.readFasta("./data/positive_0.9.fasta");
    random.shuffle(positive_data)
    # print(positive_data[215:220])
    negative_data_1 = readFasta.readFasta("./data/nega_toxin_0.9.fasta");
   # print(positive_data[1:5])
    negative_data_2 = readFasta.readFasta("./data/output_infla_0.9.fasta")[:17];
    negative_data= negative_data_1+negative_data_2
    random.shuffle(negative_data)
    # print(len(negative_data))

    all_data = negative_data[:200] + positive_data[:200]
    independent_data = negative_data[200:] + positive_data[200:]
    # print(all_data[439])
### 2.提取特征
    ### 2.1 序列特征
    ### 训练集+测试集
    name = []
    CKSAAP_2_fea = CKSAAP.CKSAAP(all_data)
    name.append(len(CKSAAP_2_fea[0]))
    CTriad_4_fea = CTriad.CTriad(all_data, 4)
    name.append(len(CTriad_4_fea[0]))
    DDE_fea = DDE.DDE(all_data)
    name.append(len(DDE_fea[0]))
    AAC_fea = AAC.AAC(all_data)
    TPC_fea = TPC.TPC(all_data)
    aac_tpc = [AAC_fea[i] + TPC_fea[i][2:] for i in range(len(AAC_fea))]
    name.append(len(aac_tpc[0]))

    CKSAAP_2_pd = pd.DataFrame(CKSAAP_2_fea, columns=CKSAAP_2_fea[0])
    CTriad_4_pd = pd.DataFrame(CTriad_4_fea, columns=CTriad_4_fea[0])
    DDE_pd = pd.DataFrame(DDE_fea, columns=DDE_fea[0])
    aac_tpc = pd.DataFrame(aac_tpc, columns=aac_tpc[0])

    # CKSAAP_2_pd = pd.DataFrame(CKSAAP_2_pd, columns=CKSAAP_2_pd[0])
    # CTriad_4_pd = pd.DataFrame(CTriad_4_pd, columns=CTriad_4_pd[0])
    # DDE_pd = pd.DataFrame(DDE_pd, columns=DDE_pd[0])
    # aac_tpc = pd.DataFrame(aac_tpc, columns=aac_tpc[0])
    ### 独立验证集
    CKSAAP_2_fea_indepedent = CKSAAP.CKSAAP(independent_data)
    CTriad_4_fea_indepedent = CTriad.CTriad(independent_data, 4)
    DDE_fea_indepedent = DDE.DDE(independent_data)
    AAC_fea_indepedent = AAC.AAC(independent_data)
    TPC_fea_indepedent = TPC.TPC(independent_data)
    aac_tpc_indepedent = [AAC_fea_indepedent[i] + TPC_fea_indepedent[i][2:] for i in range(len(AAC_fea_indepedent))]

    CKSAAP_2_pd_in = pd.DataFrame(CKSAAP_2_fea_indepedent, columns=CKSAAP_2_fea_indepedent[0])
    CTriad_4_pd_in = pd.DataFrame(CTriad_4_fea_indepedent, columns=CTriad_4_fea_indepedent[0])
    DDE_pd_in = pd.DataFrame(DDE_fea_indepedent, columns=DDE_fea_indepedent[0])
    aac_tpc_in = pd.DataFrame(aac_tpc_indepedent, columns=aac_tpc_indepedent[0])


    new_feature = CKSAAP_2_pd_in.iloc[1:,:2]
    # print(new_feature)
# # # 训练学习模型
    CKSAAP_2_pd  = machine_learning.LR_demo(CKSAAP_2_pd,CKSAAP_2_pd_in, "CKSAAP_2")
    # print(CKSAAP_2_pd[0])
    CTriad_4_pd = machine_learning.LR_demo(CTriad_4_pd,CTriad_4_pd_in,'CTriad_4')
    # print(CTriad_4_pd[0])
    DDE_pd = machine_learning.LR_demo(DDE_pd,DDE_pd_in,"DDE")
    # print(DDE_pd[0])
    aac_tpc = machine_learning.LR_demo(aac_tpc,aac_tpc_in,'aac_tpc')
    # print(aac_tpc[0])
### 筛选特征
# copy_feature = new_feature
    # new_feature["AAC_pd"] = AAC_pd[0]
    # print(new_feature)
    index_feature = ['CKSAAP_2_pd', 'CTriad_4_pd',
                 'DDE_pd',  'aac_tpc']
    for i in index_feature:
        new_feature[i] = eval(i)[0]
        # three_pd = machine_learning.svm_self(new_feature, 'three')
        # plot_roc.plot_roc_cv(three_pd[3], three_pd[4], "Combined feature", "./")
        # new_feature = copy_feature
        # del new_feature[i]
    # print(new_feature)
    # new_feature.to_csv("independent_index.csv")
    # X, y = new_feature.iloc[:, 2:].to_numpy(), new_feature.iloc[:, 1].to_numpy()
    # # X, y = np.array(X), np.array(y)
    # # print(y)
    # X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # # 初始化模型
    # svm = SVC()
    # # 查看被选中的特征
    # rfe = RFE(estimator=svm, n_features_to_select=5, step=1)
    # rfe.fit(X_train, y_train)
    # # 查看被选中的特征
    # selected_features = pd.Series(rfe.support_, index=np.array(39))
    # print("Selected features:", selected_features[selected_features == True].index.tolist())
    #
    # ### 在测试集上测试
    # y_pred = rfe.predict(X_test)
    # print(f"Accuracy after RFE: {accuracy_score(y_test, y_pred)*100:.2f}%")
    # y_scores = rfe.predict_proba(X_test)[:,1]
    # fpr, tpr, threshold = roc_curve(y_test, y_scores)
    # roc_auc = auc(fpr,tpr)
    # plt.figure()
    # plt.plot(fpr,tpr,color="darkorange",lw=2, label='ROC curve (area = %0.2f)'%roc_auc)
    # plt.plot([0,1],[0,1],color='navy',lw=2,linestyle="--")
    # plt.xlim([0.0,1.0])
    # plt.ylim([0.1,1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()
if __name__=="__main__":
    main()
