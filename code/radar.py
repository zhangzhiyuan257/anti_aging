#import matplotlib.pyplot as plt
#import pandas as pd
#import random

#def randomColor():
    #colorArr = ["1","2","3","4","5","6","7","8","9","A","B","C","D","E","F"]
    #color = "#"+"".join([random.choice(colorArr) for i in range(6)])
    #return color

#data = pd.read_csv("boxplot.csv")

#data_acc = data['acc']
#data_mcc = data["MCC"]
#data_feature = data["feature"]
#fig, ax = plt.subplots()

#fruits = data_feature

#counts = data_mcc
#bar_labels = ['red', 'blue', '_red', 'orange']
# bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
#bar_colors = [randomColor() for i in range(len(data_acc))]

#ax.bar(fruits, counts, color=bar_colors)

# 设置x轴的标签为竖直
#plt.xticks(fruits, rotation=45)
#ax.set_ylabel('MCC')
#ax.set_xlabel('Features')
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
font = {'family': 'Times New Roman',
        'size': 12,
        }
sns.set(font_scale=1.2)
plt.rc('font',family='Times New Roman')
plt.style.use('ggplot')# 使用ggplot的绘图风格
 
# 构造数据（三个模型四个维度比较）
values1= [0.945, 0.833, 0.888, 0.779,0.863,0.9]
values2= [0.963, 0.913, 0.938, 0.877,0.924,0.8942]
values3= [0.945, 0.946, 0.946, 0.895,0.95,0.945]
values4= [0.947, 0.948, 0.948, 0.898,0.95,0.947]
values5= [0.948, 0.91, 0.929, 0.862,0.921,0.932]
 
feature = ["Sensitivity","Specificity","Accuracy","MCC","Recall","F1-score"]
 
# 设置每个数据点的显示位置，在雷达图上用角度表示
angles=np.linspace(0, 2*np.pi,len(feature), endpoint=False)
angles=np.concatenate((angles,[angles[0]]))
feature = np.concatenate((feature, [feature[0]]))
 
# 绘图
fig=plt.figure(figsize=(8,8))
# 设置为极坐标格式
ax = fig.add_subplot(111, polar=True)
 
for values in [values1, values2,values3,values4,values5]:
# 拼接数据首尾，使图形中线条封闭
    values=np.concatenate((values,[values[0]]))
    # 绘制折线图
    ax.plot(angles, values, 'o-', linewidth=2)
 
for values in [values1, values2,values3,values4]:
    values=np.concatenate((values,[values[0]]))
    # 填充颜色
    ax.fill(angles, values, alpha=0.25)
    
# 设置图标上的角度划分刻度，为每个数据点处添加标签
ax.set_thetagrids(angles * 180/np.pi, feature,fontsize=14,style='italic')
# 设置雷达图的范围
ax.set_ylim(0.5,1)
# 设置雷达图的0度起始位置
ax.set_theta_zero_location('N')
# 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
ax.set_rlabel_position(270)
plt.legend(["SVM", "RF","MLP","xgBoost","LR"], loc='best')
# 添加标题
#plt.title('Comparison of classifier evaluation indicators',fontsize = 14)
# 添加网格线
plt.show()
