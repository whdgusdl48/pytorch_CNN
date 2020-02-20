import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# #특성은 4개 샘플은 총 8개 클래스는 총 0,1,2 3개
#
# print(x_train.shape)
# print(y_train.shape)
#
# #우리가 원하는것은 8*4 , 8*1을 클래스의 개수만큼 의 샘플이 존재해야하므로 8*3행렬이 필요하다
#
# y_one_hot = torch.zeros(8,3)
# y_one_hot.scatter(1,y_train.unsqueeze(1),1)
# print(y_one_hot)
#
# W = torch.zeros((4,3),requires_grad=True)
# b = torch.zeros(1,requires_grad=True)
#
# optimizer = optim.SGD([W,b], lr=0.01)
#
# tries = 1000
# for i in range(tries + 1):
#
#     hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
#
#     cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
#
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     if i % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             i, tries, cost.item()
#         ))
#
# print(W,b)

# high-level 구현

flower_train = [[5.1, 3.5, 1.4, 0.2],
                [4.9, 3.0, 1.4, 0.2],
                [5.8, 2.6, 4.0, 1.2],
                [6.7, 3.0, 5.2, 2.3],
                [5.6, 2.8, 4.9, 2.0]
                ]
speice_train = [0, 0, 1, 2, 2]
x = torch.FloatTensor(flower_train)
y = torch.LongTensor(speice_train)

y_one = torch.zeros(5,3)
y_one.scatter(1,y.unsqueeze(1),1)


W = torch.zeros((4, 3),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer2 = optim.SGD([W,b], lr = 1)
z = x.matmul(W) + b
print(z)
nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # 가설
    z = x.matmul(W) + b
    cost = F.cross_entropy(z, y)

    # cost로 H(x) 개선
    optimizer2.zero_grad()
    cost.backward()
    optimizer2.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# nn.Module로 구현

model = nn.Linear(4, 3)

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))


# 클래스로 구현

class SoftMaxClassifierModel(nn.Module):

    def __init__(self):
        super.__init__()
        self.linear = nn.Linear(4,3)

    def forward(self,x):
        self.linear(x)


model = SoftMaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr = 1)

optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))