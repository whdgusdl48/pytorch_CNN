import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
from torch import optim
digits = load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]): # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)
    plt.show()


for i in range(5):
  print(i,'번 인덱스 샘플의 레이블 : ',digits.target[i])

print(digits.data[0])
X = digits.data# 이미지. 즉, 특성 행렬
Y = digits.target # 각 이미지에 대한 레이블

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

X = torch.tensor(X , dtype=torch.float32)
Y = torch.tensor(Y , dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

losses = []

for epoch in range(1000):
  optimizer.zero_grad()
  y_pred = model(X) # forwar 연산
  loss = loss_fn(y_pred, Y)
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 1000, loss.item()
        ))

  losses.append(loss.item())

print(losses)
plt.plot(losses)
plt.show()