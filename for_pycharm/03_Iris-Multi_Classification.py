#!pip install sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 입력
df = pd.read_csv('../data/iris3.csv')

#첫 5줄을 봅니다.
df.head()

# 그래프로 확인해 봅시다.
sns.pairplot(df, hue='species');
plt.show()

# 속성을 X, 클래스를 y로 저장합니다.
X = df.iloc[:,0:3]
y = df.iloc[:,4]

# X와 y의 첫 5줄을 출력해 보겠습니다. 
print(X[0:5])
print(y[0:5])

#문자열 형태의 y값을 숫자로 바꾸어 줍니다.
encoder =  LabelEncoder()
y1 = encoder.fit_transform(y)

# 원핫 인코딩 처리를 합니다.
Y = pd.get_dummies(y)

# 원핫 인코딩 결과를 확인합니다.
print(Y[0:5])

# 모델의 설정
model = Sequential()
model.add(Dense(10,  input_dim=3, activation='relu'))
model.add(Dense(10,  input_dim=3, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
history=model.fit(X, y2, epochs=50, batch_size=5)
