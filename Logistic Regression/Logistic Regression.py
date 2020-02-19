import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2] ]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#x_Data 와 y_Data 정의를 동시해 해주고 FloatTensor를 통해 넘파이 배열로 만들어줌

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)


#가중치와 편향을 각각 벡터로 0으로 만들어준다.
optimizer = optim.SGD([W, b], lr=1)
# 경사하강법을 통해서 가중치와 편향을 가지고 학습률 1퍼센트로 설정한 후 학습을 한다.
nb_epochs = 1000
#1000번 학습
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    #가설을 시그모이드 함수를 통해서 매개변수에 Wx + b를 넣어준다.
    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()
    #비용함수 식을 적는다. 식은 로그함수를 구현
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
#학습을 한 후에 출력한다.
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
#각각의 인덱스가 0.5를 넘으면 True 안넘으면 0.5이하면 False
print(W)

#가중치를 출력한다.
print(b)

#편향을 출력한다.