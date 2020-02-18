import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


# 데이터셋 3개의 독립변수에대한 데이터값의 출력으로 y 입력은 총 3개의 x_train
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 각각 독립변수의 가중치와 편향을 0으로 초기화
hypothesis = w1*x1_train + w2*x2_train + w3*x3_train + b
# 가설 공식 독립변수당 * 가중치 + 편향
optimizer = optim.SGD([w1,w2,w3,b], lr = 1e-5)
# 경사하강법 함수호출
nb_epochs = 3000
# 반복횟수
for epoch in range(nb_epochs + 1):

    hypothesis = w1 * x1_train + w2 * x2_train + w3 * x3_train + b
    # 가설을 세워 각 독립변수 * 가중치를 구한 다음 편향을 더한다.
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))