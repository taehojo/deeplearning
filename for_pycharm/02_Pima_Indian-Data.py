# 필요한 라이브러리를 불러옵니다.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 피마 인디언 당뇨병 데이터셋을 불러옵니다. 
df = pd.read_csv('../data/pima-indians-diabetes3.csv')

# 처음 5줄을 봅니다.
df.head(5)

# 정상과 당뇨 환자가 각각 몇명씩인지 조사해 봅니다.
df["diabetes "].value_counts()

# 각 정보별 특징을 좀더 자세히 출력합니다.
df.describe()

# 각 항목이 어느정도의 상관관계를 가지고 있는지 알아봅니다. 
df.corr()

# 데이터 간의 상관관계를 그래프로 표현해 봅니다.
colormap = plt.cm.gist_heat   #그래프의 색상 구성을 정합니다.
plt.figure(figsize=(12,12))   #그래프의 크기를 정합니다.

# 그래프의 속성을 결정합니다. vmax의 값을 0.5로 지정해 0.5에 가까울 수록 밝은 색으로 표시되게 합니다.
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()

#plasma를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포하는지 살펴 봅니다. 
plt.hist(x=[df.plasma[df.diabetes ==0], df.plasma[df.diabetes ==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()


#BMI를 기준으로 각각 정상과 당뇨가 어느 정도 비율로 분포하는지 살펴 봅니다. 
plt.hist(x=[df.bmi[df.diabetes ==0], df.bmi[df.diabetes ==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()