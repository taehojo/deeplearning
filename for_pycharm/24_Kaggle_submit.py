from tensorflow.keras.models import load_model
import pandas as pd

#캐글에서 다운로드 받은 테스트셋을 불러옵니다.
kaggle_test = pd.read_csv("../data/house_test.csv")

#카테고리형 변수를 0과 1로 이루어진 변수로 바꾸어 줍니다.(12장 3절)
kaggle_test = pd.get_dummies(kaggle_test)

#결측치를 전체 칼럼의 평균으로 대체하여 채워줍니다. 
kaggle_test = kaggle_test.fillna(kaggle_test.mean())

#학습에 사용된 열을 저장합니다. 
cols_kaggle=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
K_test = kaggle_test[cols_kaggle]

#앞서 15장에서 만든 모델을 불러 옵니다.
model = load_model("../data/model/Ch15-house.hdf5")

# 예측값과 실제값, 실행 번호가 들어갈 빈 리스트를 만듭니다.
ids =[]

# 25개의 샘플을 뽑아 실제값, 예측값을 출력해 봅니다. 
Y_prediction = model.predict(K_test).flatten()
for i in range(len(K_test)):
    id = kaggle_test['Id'][i]
    prediction = Y_prediction[i]
    ids.append([id, prediction])

# 테스트 결과의 저장환경을 설정합니다.
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
filename = str(timestr)
outdir = '../data/kaggle/' 

#Id와 집값을 csv파일로 저장합니다.
df = pd.DataFrame(ids, columns=["Id", "SalePrice"])
df.to_csv(str(outdir + filename + '_submission.csv'), index=False)


