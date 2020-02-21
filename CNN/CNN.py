import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import matplotlib.pyplot as plt
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size= 3, stride = 1, padding= 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 첫번째 층은 32채널로 만들고 커널사이즈를 3으로 준다 즉 3*3형태 스트라이드는 1로 준다 간격을 1로 한다 . padding을 1로준다.
        # ReLu함수를 통해 연결 MaxPool2d를 통해 최대풀링을 해준다 커널 사이즈를 2로 하고 스트라이드를 2로해준다.
        # 현재까지 이미지는 28,28,1에서 28,28,32로 변하고 풀링을 통해 14,14,32로 되었다.

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 두번째 층은 64채널로 만들고 똑같이 커널 사이즈를 3으로하고 스트라이드를 1로 준다.
        # ReLu함수를 구통해 연결한 후 MaxPool2d를 통해 최대 풀링을 해준다 커널 사이즈를 2로하고 스트라이드를 2로해준다.
        # 여기까지 2번째 층을 통과하게 된다면  14,14,32 에서 14,14,64로 변하고 7,7,64로 해준다.
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias= True)
        # 전결합층 함수를 통해 10개로 출력값을 10개로 줄인다. 이것도 결국 신경망의 일종이다.

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out


model = CNN().to(device)
#CNN클래스를 디바이스에 연결한 값을 모델로 설정한 후
criterion = torch.nn.CrossEntropyLoss().to(device)
# 비용함수를 구현한다. 이 함수는 소프트맥스함수다.
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
#Adam 함수를 통해 최적의 값을 찾아 내려간다.
total_batch = len(data_loader)
print(total_batch)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

    print((epoch + 1, avg_cost))
    avg_cost += cost / total_batch

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    r = random.randint(0, len(mnist_test) - 1)
    print('Accuracy:', accuracy.item())
    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()