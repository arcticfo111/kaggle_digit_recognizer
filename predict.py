import numpy as np # 숫자 관련
import pandas as pd # 데이터 분석, 데이터셋, 처리 등
import matplotlib.pyplot as plt # 그래프 관련

import torch # 딥러닝 알고리즘, 텐서 연산 넘파이와 유사하지만 gpu지원 

# config 설정
class PATH:
    TRAIN = './data/train.csv'
    TEST = './data/test.csv'
    
class CONFIG:
    lr = 0.001
    epoch = 10
    batch_size = 256

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
