import numpy as np
import matplotlib.pyplot as plt

#텐서플로의 케라스 API에서 필요한 함수들을 불러 옵니다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97]) 

model = Sequential()

# y=ax+b에서 출력 값 (y)이 하나이므로 Dense(1), 입력 변수(x)가 하나이므로 input_dim=1, 선형 회귀 모델이므로 activation 옵션에 'linear' 선택
model.add(Dense(1, input_dim=1, activation='linear'))

# 오차 수정을 위해 경사하강법(sgd)을, 오차의 정도를 판단하기 위해 평균 제곱 오차(mse)를 사용
model.compile(optimizer='sgd', loss='mse')

# 오차를 최소화하는 과정을 2000번 반복
model.fit(x, y, epochs=2000)

plt.scatter(x, y)
plt.plot(x, model.predict(x),'r')    # 예측 결과를 그래프로 나타내기
plt.show()

#임의의 시간을 집어넣어 점수를 예측하는 모델을 테스트해 보겠습니다.
hour = 7
prediction = model.predict([hour])
print("%.f시간을 공부할 경우의 예상 점수는 %.02f점입니다" % (hour, prediction))