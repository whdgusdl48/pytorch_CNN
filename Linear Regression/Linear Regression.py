import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
#x_train은 입력
#y_train은 출력이라는 가설을 세운다.
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)

# 실험데이터에 대한 입력 출력 정의 DataSet이 되겠음

W = torch.zeros(1,requires_grad=True)
print(W)

b = torch.zeros(1,requires_grad=True)
print(b)

# 가중치 편향 초기값 설정 즉 y=Wx + b => 0x + 0

hypothesis = x_train * W + b
print(hypothesis)

cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)
# 비용함수 선언방식 가설값에 출력값을 뺀 값을 제곱하고 평균을 구한다. 선형회귀방식
optimizer = optim.SGD([W, b], lr=0.01)
# 선형회귀법에서 경사 하강법이 가장 적절하므로 optimizer를 선언한후 SGD 함수를 통해 가중치와 편향을 구한다.
nb_epochs = 5000 # 원하는만큼 경사 하강법을 반복
for epoch in range(nb_epochs + 1):
    # 반복문을 돌려 가설을 세운 후
    hypothesis = x_train * W + b

    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    # 미분을 통해 얻은 기울기를 0으로 초기화 이래야만 새로운 가중치 편향에 대한 새로운 기울기를 얻을 수 있음
    cost.backward()
    # 가중치와 편향이 계산된값이 나옴
    optimizer.step()
    # 학습률를 0.01을 곱하여 빼줘서업데이트해줌

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
        # 100번마다 출력

# 자동미분 공식
w2 = torch.tensor(2.0,requires_grad=True)

y2 = w2**2
z2 = 2*y2 + 5
# 식 y = 2(x*x) + 5
z2.backward()
# 자동미분 함수

print(format(w2.grad))
# 자동미분식완성