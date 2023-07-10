import numpy as np # 숫자 관련
import pandas as pd # 데이터 분석, 데이터셋, 처리 등
import matplotlib.pyplot as plt # 그래프 관련

import torch # 딥러닝 알고리즘, 텐서 연산 넘파이와 유사하지만 gpu지원 
import torch.nn as nn # pytorch 신경망 구성에 필요한 클래스와 함수가 포함된 모듈
import torch.nn.functional as F # pytorch 함수버전의 신경망 연산에 접근 할 수 있는 모듈
from torch.utils.data import Dataset,DataLoader # 파이토치에서 사용자 정의 데이터셋 생성(이 클래스를 상속받아서 )

from models.Conv2d import Conv2d

# config 설정
class PATH:
    TRAIN = './data/train.csv'
    TEST = './data/test.csv'
    
class CONFIG:
    lr = 0.001
    epoch = 10
    batch_size = 256


# gpu 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 학습 데이터 만들기
df_train = pd.read_csv(PATH.TRAIN)
l = len(df_train) # 데이터셋 행 갯수

# 데이터 프레임 형태인 이미지를 변환하는 전처리
## 28*28의 형태로 변환후 0~1 사이의 값으로 변환
_x = torch.Tensor(df_train.iloc[:,1:].values).reshape(l,28,28) / 256 
_x = _x.unsqueeze(1) # (batch_size, depth, width, height)로 형태를 맞추기 위한 코드
_y = torch.Tensor(df_train.iloc[:,0].values).type(torch.long)
## 훈련과 검증 셋으로 분할하기
train_x ,valid_x = _x[:39000],_x[39000:]
train_y ,valid_y = _y[:39000],_y[39000:]

# torch.utils.data의 Dataset과 DataLoader를 이용하여 데이터를 불러오고, shuffle하거나, batch별로 불러오기
class DigitDataset(Dataset):
    
    def __init__(self,x,y=None):
        super(DigitDataset).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self,idx):
        if self.y == None:
            return self.x[idx]
        return self.x[idx],self.y[idx]
    
    def __len__(self):
        return len(self.x)
trainset = DigitDataset(train_x,train_y)
validset = DigitDataset(valid_x,valid_y)
train_dl = DataLoader(trainset,batch_size = CONFIG.batch_size,shuffle=True)
valid_dl = DataLoader(validset,batch_size = CONFIG.batch_size)

# 학습 모델 실행
model = Conv2d().to(device)
lossfn = nn.CrossEntropyLoss() # 손실함수
opt = torch.optim.Adam(model.parameters(), lr=CONFIG.lr) # optimizer

## 한 번의 에포크에 대해 trainset 전체를 한 번 학습사고, validset으로 검증
for e in range(CONFIG.epoch):
    print(f"Training on progress... {e+1}/{CONFIG.epoch}")
    acc =0
    total =0
    
    model.train() # train
    for x,y in train_dl:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = lossfn(y_hat,y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    model.eval() #valid
    with torch.no_grad():
        for vx,vy in valid_dl:
            vx = vx.to(device)
            vy = vy.to(device)
            
            vy_hat = model(vx)
            pred = vy_hat.max(dim=1)[1]
            acc += (pred == vy).sum().item()
    print(f"Epoch {e+1} : Acc {100*acc/3000}")

# predict 
df_test= pd.read_csv(PATH.TEST)
l = len(df_test)
test_x = torch.Tensor(df_test.values).reshape(l,28,28) / 256
test_x = test_x.unsqueeze(1)

outputs = model(test_x)
_, pred = torch.max(outputs, 1)
pred = pred.cpu()

for i,img in enumerate(test_x[50:56]) :
    plt.subplot(2,3,i+1)
    plt.axis('off')
    plt.title(f"Predicted : {pred[50+i]}")
    plt.imshow(img.squeeze(0))

# 제출 파일 생성
submission = pd.DataFrame({'ImageId': np.arange(1, (pred.size(0) + 1)), 'Label': pred})
submission.to_csv("submission.csv", index = False)
