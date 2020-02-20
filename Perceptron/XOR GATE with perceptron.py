import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# cuda를 통해 구현한다.
X = torch.FloatTensor([[0, 0],[0, 1], [1, 0] , [1 ,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# X인자로는 (0,0) (0,1) (1,0) (1,1) 을통해 좌표값을 device에 보낸다.
# Y인자로는 실제 값을 비교하기위해 설정 XOR
linear = nn.Linear(2, 1, bias=True)
# 4,2 4,1 => 2,1
sigmoid = nn.Sigmoid()
# 시그모이드 함수 구현
model = nn.Sequential(linear, sigmoid).to(device)
# 단층 퍼셉트론을 구현하기위해 다중클래스 구별법인 소프트맥스 회귀함수를 가지고 모델로 설ㅈㅇ
criterion = torch.nn.BCELoss().to(device)
# 이진분류에 사용되는 크로스 엔트로피 함수
optimizer = optim.SGD(model.parameters(), lr = 1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    # 비용 함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0: # 100번째 에포크마다 비용 출력
        print(step, cost.item())
    if step % 100 == 0: # 100번째 에포크마다 비용 출력
        print(step, cost.item())
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())