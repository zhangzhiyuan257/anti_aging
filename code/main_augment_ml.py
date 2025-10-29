import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_extraction import *
from other_script import *
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report,roc_curve,auc
import random as rm
# def data_augment(file_name):
#     f = pd.read_csv('Training.csv')

def main():
    ###################
    ##### 数据读取 #####
    ##################
    data_posittive = pd.read_csv("../data/posi_8.csv")
    data_negative = pd.read_csv("../data/nega_8.csv")
    print(len(data_negative))
    all_data = data_posittive + data_negative
    DDE_feature_neg = DDE(data_negative)
    CKSAAP_feature_neg = CKSAAP(data_negative)
    CTriad_feature_neg = CTriad(data_negative)
    ####################
    ##### 数据处理 ######
    ###################
    DDE_feature_neg = DDE_feature_neg[1:]
    CKSAAP_feature_neg = CKSAAP_feature_neg[1:]
    CTriad_feature_neg = CTriad_feature_neg[1:]

    data_list_dde = [i[2:] for i in DDE_feature_neg]
    data_list_cks = [i[2:] for i in CKSAAP_feature_neg]
    data_list_ct = [i[2:] for i in CTriad_feature_neg]
    if model_1 == True:
        data_all = [data_list_dde[i] + data_list_cks[i] + data_list_ct[i] for i in range(len(data_list_dde))]
        data_all = np.array(data_all)
        print(len(data_all), len(data_all[0]))
    data_list_dde = np.array(data_list_dde)
    data_list_cks = np.array(data_list_cks)
    data_list_ct = np.array(data_list_ct)

### 2.提取特征
    ### 2.1 序列特征
    name = []
    AAC_fea = AAC.AAC(all_data)
    name.append(len(AAC_fea[0]))
    #
    AAC_count_fea = AAC_count.AAC(all_data)
    name.append(len(AAC_count_fea[0]))

    # AAindex_fea = AAINDEX.AAINDEX(all_data)
    # AAindex_fea = [i[:5000] for i in AAindex_fea]
    # name.append(len(AAindex_fea[0]))

    CKSAAP_2_fea = CKSAAP.CKSAAP(all_data)
    name.append(len(CKSAAP_2_fea[0]))

    CKSAAP_3_fea = CKSAAP.CKSAAP(all_data, 3)
    name.append(len(CKSAAP_3_fea[0]))

    CKSAAP_4_fea = CKSAAP.CKSAAP(all_data, 4)
    name.append(len(CKSAAP_4_fea[0]))

    CKSAAP_5_fea = CKSAAP.CKSAAP(all_data, 5)
    name.append(len(CKSAAP_5_fea[0]))

    CKSAAP_6_fea = CKSAAP.CKSAAP(all_data, 6)
    name.append(len(CKSAAP_6_fea[0]))

    CTDC_fea = CTDC.CTDC(all_data)
    name.append(len(CTDC_fea[0]))

    CTDD_fea = CTDD.CTDD(all_data)
    name.append(len(CTDD_fea[0]))

    CTDT_fea = CTDT.CTDT(all_data)
    name.append(len(CTDT_fea[0]))

    CTriad_0_fea = CTriad.CTriad(all_data)
    name.append(len(CTriad_0_fea[0]))

    CTriad_1_fea = CTriad.CTriad(all_data, 1)
    name.append(len(CTriad_1_fea[0]))

    CTriad_2_fea = CTriad.CTriad(all_data, 2)
    name.append(len(CTriad_2_fea[0]))

    CTriad_3_fea = CTriad.CTriad(all_data, 3)
    name.append(len(CTriad_3_fea[0]))

    CTriad_4_fea = CTriad.CTriad(all_data, 4)
    name.append(len(CTriad_4_fea[0]))

    CTriad_5_fea = CTriad.CTriad(all_data, 5)
    name.append(len(CTriad_5_fea[0]))

    PAAC_fea = PAAC.PAAC(all_data)
    name.append(len(PAAC_fea[0]))

    DDE_fea = DDE.DDE(all_data)
    name.append(len(DDE_fea[0]))

    DPC_fea = DPC.DPC(all_data)
    name.append(len(DPC_fea[0]))

    EAAC_5_fea = EAAC.EAAC(all_data)
    name.append(len(EAAC_5_fea[0]))

    EAAC_6_fea = EAAC.EAAC(all_data, 6)
    name.append(len(EAAC_6_fea[0]))

    EAAC_7_fea = EAAC.EAAC(all_data, 7)
    name.append(len(EAAC_7_fea[0]))

    EAAC_8_fea = EAAC.EAAC(all_data, 8)
    name.append(len(EAAC_8_fea[0]))

    EAAC_9_fea = EAAC.EAAC(all_data, 9)
    name.append(len(EAAC_9_fea[0]))

    EAAC_10_fea = EAAC.EAAC(all_data, 10)
    name.append(len(EAAC_10_fea[0]))

    p_EAAC_5_fea = EAAC_pencent.EAAC(all_data)
    name.append(len(p_EAAC_5_fea[0]))

    p_EAAC_6_fea = EAAC_pencent.EAAC(all_data, 6)
    name.append(len(p_EAAC_6_fea[0]))

    p_EAAC_7_fea = EAAC_pencent.EAAC(all_data, 7)
    name.append(len(p_EAAC_7_fea[0]))

    p_EAAC_8_fea = EAAC_pencent.EAAC(all_data, 8)
    name.append(len(p_EAAC_8_fea[0]))

    p_EAAC_9_fea = EAAC_pencent.EAAC(all_data, 9)
    name.append(len(p_EAAC_9_fea[0]))

    p_EAAC_10_fea = EAAC_pencent.EAAC(all_data, 10)
    name.append(len(p_EAAC_10_fea[0]))

    GAAC_fea = GAAC.GAAC(all_data)
    name.append(len(GAAC_fea[0]))

    EGAAC_5_fea = EGAAC.EGAAC(all_data)
    name.append(len(EGAAC_5_fea[0]))

    # EGAAC_6_fea = EGAAC.EGAAC(all_data, 6)
    # name.append(len(EGAAC_6_fea[0]))
    #
    # EGAAC_7_fea = EGAAC.EGAAC(all_data, 7)
    # name.append(len(EGAAC_7_fea[0]))
    #
    # EGAAC_8_fea = EGAAC.EGAAC(all_data, 8)
    # name.append(len(EGAAC_8_fea[0]))
    #
    # EGAAC_9_fea = EGAAC.EGAAC(all_data, 9)
    # name.append(len(EGAAC_9_fea[0]))
    #
    # EGAAC_10_fea = EGAAC.EGAAC(all_data, 10)
    # name.append(len(EGAAC_10_fea[0]))

    GDPC_fea = GDPC.GDPC(all_data)
    name.append(len(GDPC_fea[0]))

    Geary_fea = Geary.Geary(all_data)
    name.append(len(Geary_fea[0]))

    GTPC_fea = GTPC.GTPC(all_data)
    name.append(len(GTPC_fea[0]))

    TPC_fea = TPC.TPC(all_data)
    name.append(len(TPC_fea[0]))

    aac_dpc = [AAC_fea[i] + DPC_fea[i][2:] for i in range(len(AAC_fea))]
    name.append(len(aac_dpc[0]))
    aac_tpc = [AAC_fea[i] + TPC_fea[i][2:] for i in range(len(AAC_fea))]
    name.append(len(aac_tpc[0]))
    ctd = [CTDC_fea[i] + CTDD_fea[i][2:] + CTDT_fea[i][2:] for i in range(len(CTDC_fea))]
    name.append(len(ctd[0]))

    # # # 转化为pandas格式
    # #
    # AAindex_pd = pd.DataFrame(AAindex_fea, columns=AAindex_fea[0])

    AAC_count_pd = pd.DataFrame(AAC_count_fea, columns=AAC_count_fea[0])
    AAC_pd = pd.DataFrame(AAC_fea, columns=AAC_fea[0])
    # print(AAC_pd)
    CTDC_pd = pd.DataFrame(CTDC_fea, columns=CTDC_fea[0])
    CTDT_pd = pd.DataFrame(CTDT_fea, columns=CTDT_fea[0])
    CKSAAP_2_pd = pd.DataFrame(CKSAAP_2_fea, columns=CKSAAP_2_fea[0])
    CKSAAP_3_pd = pd.DataFrame(CKSAAP_3_fea, columns=CKSAAP_3_fea[0])
    CKSAAP_4_pd = pd.DataFrame(CKSAAP_4_fea, columns=CKSAAP_4_fea[0])
    CKSAAP_5_pd = pd.DataFrame(CKSAAP_5_fea, columns=CKSAAP_5_fea[0])
    CKSAAP_6_pd = pd.DataFrame(CKSAAP_6_fea, columns=CKSAAP_6_fea[0])
    GAAC_pd = pd.DataFrame(GAAC_fea, columns=GAAC_fea[0])
    CTriad_0_pd = pd.DataFrame(CTriad_0_fea, columns=CTriad_0_fea[0])
    CTriad_1_pd = pd.DataFrame(CTriad_1_fea, columns=CTriad_1_fea[0])
    CTriad_2_pd = pd.DataFrame(CTriad_2_fea, columns=CTriad_2_fea[0])
    CTriad_3_pd = pd.DataFrame(CTriad_3_fea, columns=CTriad_3_fea[0])
    CTriad_4_pd = pd.DataFrame(CTriad_4_fea, columns=CTriad_4_fea[0])
    CTriad_5_pd = pd.DataFrame(CTriad_5_fea, columns=CTriad_5_fea[0])
    PAAC_pd = pd.DataFrame(PAAC_fea, columns=PAAC_fea[0])
    DDE_pd = pd.DataFrame(DDE_fea, columns=DDE_fea[0])
    DPC_pd = pd.DataFrame(DPC_fea, columns=DPC_fea[0])
    EAAC_5_pd = pd.DataFrame(EAAC_5_fea, columns=EAAC_5_fea[0])
    EAAC_6_pd = pd.DataFrame(EAAC_6_fea, columns=EAAC_6_fea[0])
    EAAC_7_pd = pd.DataFrame(EAAC_7_fea, columns=EAAC_7_fea[0])
    EAAC_8_pd = pd.DataFrame(EAAC_8_fea, columns=EAAC_8_fea[0])
    EAAC_9_pd = pd.DataFrame(EAAC_9_fea, columns=EAAC_9_fea[0])
    EAAC_10_pd = pd.DataFrame(EAAC_10_fea, columns=EAAC_10_fea[0])
    p_EAAC_5_pd = pd.DataFrame(p_EAAC_5_fea, columns=p_EAAC_5_fea[0])
    p_EAAC_6_pd = pd.DataFrame(p_EAAC_6_fea, columns=p_EAAC_6_fea[0])
    p_EAAC_7_pd = pd.DataFrame(p_EAAC_7_fea, columns=p_EAAC_7_fea[0])
    p_EAAC_8_pd = pd.DataFrame(p_EAAC_8_fea, columns=p_EAAC_8_fea[0])
    p_EAAC_9_pd = pd.DataFrame(p_EAAC_9_fea, columns=p_EAAC_9_fea[0])
    p_EAAC_10_pd = pd.DataFrame(p_EAAC_10_fea, columns=p_EAAC_10_fea[0])
    EGAAC_5_pd = pd.DataFrame(EGAAC_5_fea, columns=EGAAC_5_fea[0])
    # EGAAC_6_pd = pd.DataFrame(EGAAC_6_fea, columns=EGAAC_6_fea[0])
    # EGAAC_7_pd = pd.DataFrame(EGAAC_7_fea, columns=EGAAC_7_fea[0])
    # EGAAC_8_pd = pd.DataFrame(EGAAC_8_fea, columns=EGAAC_8_fea[0])
    # EGAAC_9_pd = pd.DataFrame(EGAAC_9_fea, columns=EGAAC_9_fea[0])
    # EGAAC_10_pd = pd.DataFrame(EGAAC_10_fea, columns=EGAAC_10_fea[0])
    GDPC_pd = pd.DataFrame(GDPC_fea, columns=GDPC_fea[0])
    # Geary_pd = pd.DataFrame(Geary_fea, columns=Geary_fea[0])
    TPC_pd = pd.DataFrame(TPC_fea, columns=TPC_fea[0])
    aac_dpc = pd.DataFrame(aac_dpc, columns=aac_dpc[0])
    aac_tpc = pd.DataFrame(aac_tpc, columns=aac_tpc[0])
    ctd = pd.DataFrame(ctd, columns=ctd[0])
    CTDD_pd = pd.DataFrame(CTDD_fea, columns=CTDD_fea[0])
    GTPC_pd = pd.DataFrame(GTPC_fea, columns=GTPC_fea[0])

    new_feature = AAC_pd.iloc[1:,:2]
    # print(new_feature)
# # # 训练学习模型
    AAC_pd  = machine_learning.svm_self(AAC_pd,"AAC")
    # AAindex_pd  = machine_learning.svm_self(AAindex_pd,"AAindex")
    CKSAAP_2_pd  = machine_learning.svm_self(CKSAAP_2_pd,"CKSAAP_2")
    CKSAAP_3_pd, CKSAAP_4_pd, CKSAAP_5_pd,CKSAAP_6_pd = machine_learning.svm_self(CKSAAP_3_pd,"CKSAAP_3"), machine_learning.svm_self(CKSAAP_4_pd,'CKSAAP_4'), machine_learning.svm_self(CKSAAP_5_pd,'CKSAAP_5'), machine_learning.svm_self(CKSAAP_6_pd,'CKSAAP_6')
    CTriad_0_pd, CTriad_1_pd, CTriad_2_pd = machine_learning.svm_self(CTriad_0_pd,"CTriad_0"), machine_learning.svm_self(CTriad_1_pd,'CTriad_1'), machine_learning.svm_self(CTriad_2_pd,'CTriad_2')
    CTriad_3_pd, CTriad_4_pd, CTriad_5_pd = machine_learning.svm_self(CTriad_3_pd,"CTriad_3"), machine_learning.svm_self(CTriad_4_pd,'CTriad_4'), machine_learning.svm_self(CTriad_5_pd,'CTriad_5')
    DDE_pd, DPC_pd = machine_learning.svm_self(DDE_pd,"DDE"), machine_learning.svm_self(DPC_pd,'DPC')
    CTDC_pd, CTDD_pd =  machine_learning.svm_self(CTDC_pd,'CTDC'), machine_learning.svm_self(CTDD_pd,'CTDD')

    CTDT_pd, AAC_count_pd, PAAC_pd = machine_learning.svm_self(CTDT_pd,'CTDT'), machine_learning.svm_self(AAC_count_pd,'AAC_count'), machine_learning.svm_self(PAAC_pd,'PAAC')
    # CTriad_0_pd = machine_learning.ml(CTriad_0_pd,"CTriad")
    aac_dpc, aac_tpc, ctd = machine_learning.svm_self(aac_dpc,'aac_dpc'), machine_learning.svm_self(aac_tpc,'aac_tpc'), machine_learning.svm_self(ctd,'ctd')
    #
    EAAC_7_pd,EAAC_5_pd, EAAC_6_pd = machine_learning.svm_self(EAAC_7_pd,"EAAC_7"), machine_learning.svm_self(EAAC_5_pd,'EAAC_5'), machine_learning.svm_self(EAAC_6_pd,'EAAC_6')
    EAAC_8_pd, EAAC_9_pd, EAAC_10_pd = machine_learning.svm_self(EAAC_8_pd,"EAAC_8"), machine_learning.svm_self(EAAC_9_pd,'EAAC_9'), machine_learning.svm_self(EAAC_10_pd,'EAAC_10')

    p_EAAC_7_pd,p_EAAC_5_pd, p_EAAC_6_pd = machine_learning.svm_self(p_EAAC_7_pd,"p_EAAC_7"), machine_learning.svm_self(p_EAAC_5_pd,'p_EAAC_5'), machine_learning.svm_self(p_EAAC_6_pd,'p_EAAC_6')
    p_EAAC_8_pd, p_EAAC_9_pd, p_EAAC_10_pd = machine_learning.svm_self(p_EAAC_8_pd,"EAAC"), machine_learning.svm_self(p_EAAC_9_pd,'EAAC'), machine_learning.svm_self(p_EAAC_10_pd,'p_EAAC_10')
    EGAAC_5_pd=  machine_learning.svm_self(EGAAC_5_pd,'EGAAC_5')
    # EGAAC_8_pd, EGAAC_9_pd, EGAAC_10_pd = machine_learning.svm_self(EGAAC_8_pd,"EGAAC_8"), machine_learning.svm_self(EGAAC_9_pd,'EGAAC_9'), machine_learning.svm_self(EGAAC_10_pd,'EGAAC_10')

    GDPC_pd = machine_learning.svm_self(GDPC_pd, "GDPC")
    # Geary_pd= machine_learning.svm_self(Geary_pd,'Geary')
    GTPC_pd =  machine_learning.svm_self(GTPC_pd,'GTPC')
    TPC_pd=machine_learning.svm_self(TPC_pd,'TPC')
    GAAC_pd = machine_learning.svm_self(GAAC_pd,"GAAC")

### 筛选特征
# copy_feature = new_feature
    # new_feature["AAC_pd"] = AAC_pd[0]
    # print(new_feature)
    index_feature = ['AAC_pd', 'CKSAAP_2_pd', 'CKSAAP_3_pd', 'CKSAAP_4_pd', 'CKSAAP_5_pd', 'CKSAAP_6_pd', 'CTriad_0_pd',
                 'CTriad_1_pd', 'CTriad_2_pd', 'CTriad_3_pd', 'CTriad_4_pd', 'CTriad_5_pd',
                 'DDE_pd', 'DPC_pd', 'CTDC_pd', 'CTDD_pd', 'CTDT_pd', 'AAC_count_pd', 'PAAC_pd', 'aac_dpc', 'aac_tpc', 'ctd', 'EAAC_7_pd',
                 'EAAC_5_pd', 'EAAC_6_pd', 'EAAC_8_pd', 'EAAC_9_pd', 'EAAC_10_pd', 'p_EAAC_7_pd',
                 'p_EAAC_5_pd', 'p_EAAC_6_pd', 'p_EAAC_8_pd', 'p_EAAC_9_pd', 'p_EAAC_10_pd', 'EGAAC_5_pd',
                'GDPC_pd',  'GTPC_pd','TPC_pd', 'GAAC_pd' ]
    for i in index_feature:
        new_feature[i] = eval(i)[0]
        # three_pd = machine_learning.svm_self(new_feature, 'three')
        # plot_roc.plot_roc_cv(three_pd[3], three_pd[4], "Combined feature", "./")
        # new_feature = copy_feature
        # del new_feature[i]
    # print(new_feature)
    new_feature.to_csv("index.csv")
    X, y = new_feature.iloc[:, 2:].to_numpy(), new_feature.iloc[:, 1].to_numpy()

    # X, y = np.array(X), np.array(y)

    X_train,  X_test, y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(X_train)
    print(y_train)
    # 初始化模型
    svm = SVC()
    # 查看被选中的特征
    rfe = RFE(estimator=svm, n_features_to_select=5, step=1)
    rfe.fit(X_train, y_train)
    # 查看被选中的特征
    selected_features = pd.Series(rfe.support_, index=np.array(39))
    print("Selected features:", selected_features[selected_features == True].index.tolist())

    ### 在测试集上测试
    y_pred = rfe.predict(X_test)
    print(f"Accuracy after RFE: {accuracy_score(y_test, y_pred)*100:.2f}%")
    y_scores = rfe.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr,tpr,color="darkorange",lw=2, label='ROC curve (area = %0.2f)'%roc_auc)
    plt.plot([0,1],[0,1],color='navy',lw=2,linestyle="--")
    plt.xlim([0.0,1.0])
    plt.ylim([0.1,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
if __name__=="__main__":
    main()