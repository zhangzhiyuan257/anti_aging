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
    train_data = pd.read_csv("index_fold.csv")
    validated_data = pd.read_csv("independent_index.csv")
    # print(all_data[439])

    # print(new_feature)
# # # 训练学习模型
    fold_4_feature  = machine_learning.LR_demo(train_data,validated_data, "fold")
    # print(CKSAAP_2_pd[0])


if __name__=="__main__":
    main()
