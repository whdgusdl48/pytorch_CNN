import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
#x_Data 와 y_Data 정의를 동시해 해주고 FloatTensor를 통해 넘파이 배열로 만들어줌
model = nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

# 모델 정의 2,1 행렬을 선형적으로 선언하고 시그모이드 함수를 호출한다.

optimizer = optim.SGD(model.parameters(), lr = 1)

# 경사하강법을 구현한다. model.parameter로 받는다.
tries = 1000

for i in range(tries + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if i % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])  # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
            i, tries, cost.item(), accuracy * 100,
        ))

model(x_train)
pltshow = list(model.parameters())
