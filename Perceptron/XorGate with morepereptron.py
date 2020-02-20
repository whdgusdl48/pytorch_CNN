import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

X = torch.FloatTensor([[0, 0],[0, 1], [1, 0] , [1 ,1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
# 단층과 유사 그러나 모델 구현법에서 차이가난다.
model = nn.Sequential(
    nn.Linear(2,8,bias=True),
    nn.Sigmoid(),
    nn.Linear(8, 8, bias=True),

    nn.Sigmoid(),
    nn.Linear(8 ,8 ,bias=True),

    nn.Sigmoid(),
    nn.Linear(8, 1, bias=True),
    nn.Sigmoid()).to(device)
# 2개의 입력값과 8개의 은닉창이 2개를 구현한 후 1개의 출력층을 구현하는 모델을 구현

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)
# 단층과 유사
tries = 10000
for i in range(tries + 1):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()

    if i % 100 == 0:
        print(i, cost.item())

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())