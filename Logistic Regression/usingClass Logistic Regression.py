import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = BinaryClassifier()
# 모델클래스를 구현한다. 선형과 시그모이드 함수를 선언한 후 에 객체 생성을 통해 구현한다.
optimizer = optim.SGD(model.parameters(),lr = 1)
tries = 1000
for i in range(tries + 1):

    hypothesis = model(x_train)

    cost = F.binary_cross_entropy(hypothesis,y_train)
    #파이토치 라이브러리 비용함수 구현
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if i % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])  # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
            i, tries, cost.item(), accuracy * 100,
        ))

model(x_train)

print(list(model.parameters()))