
import torch # 딥러닝 알고리즘, 텐서 연산 넘파이와 유사하지만 gpu지원 
import torch.nn as nn # pytorch 신경망 구성에 필요한 클래스와 함수가 포함된 모듈

class Conv2d(nn.Module):
    
    def __init__(self):
        super(Conv2d,self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.Conv2d(32,8,3),
            nn.Flatten(),
            nn.Linear(24*24*8,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
        
    def forward(self,x):
        x = self.model(x)
        x = torch.softmax(x,dim=-1)
        return x