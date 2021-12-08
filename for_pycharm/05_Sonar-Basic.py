from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd

# 데이터 입력
df = pd.read_csv('../data/sonar3.csv', header=None)

# 첫 5줄을 봅니다. 
df.head()

# 일반 암석(R)과 광석(M)이 몇개가 있는지 확인합니다.
df[60].value_counts()

# 음파 관련 속성을 X로, 광물의 종류를 y로 저장합니다.
X = df.iloc[:,0:60]
Y = df.iloc[:,60]

# 모델을 설정합니다.
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델을 실행합니다.
history=model.fit(X, Y, epochs=200, batch_size=10)