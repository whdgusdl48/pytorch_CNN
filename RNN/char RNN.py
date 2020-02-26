import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#데이터 전처리
input_str = 'apple'
label_str = 'pple!'

#입력값은 apple 라벨값은 pple!
char_vocab = sorted(list(set(input_str+label_str)))
voca_size = len(char_vocab)

#문자 단어장을 sort를 통해서 list형태로 구현 중복을 피하게하기위해서 set함수선언

input_size = voca_size

# a p l e ! 이렇게 5개가 있다.
hiddensize = 5
# 은닉층의 크기를 입력층의 크기랑 같게한다.
outputsize = 5
# 출력도 5로 한다.
lr = 0.1
#학습률은 0.1

charindex = dict((c,i) for i,c in enumerate(char_vocab))
#문자 인덱스는 사전형태로 정의하여 각 문자열의 고유 정수를 부여한다. 

index_to_char={}
for key, value in charindex.items():
    index_to_char[value] = key

# 숫자와 문자를 서로 바꿔줌
x_data = [charindex[c] for c in input_str]
# apple에 해당되는 숫자의 값을 가진다.
y_data = [charindex[c] for c in label_str]
# pple!에 해당되는 숫자의 값을 가진다.
print(x_data)
print(y_data)
x_data = [x_data]
y_data = [y_data]
#파이토치 텐서형태로 하나의 차원을 추가한다.

x_one_hot = [np.eye(voca_size)[x] for x in x_data]
#원핫 벡터로 eye함수를 통해 위치값에 x의 크기 * x의 크기 행렬에 값을 넣어준다
print(x_one_hot)

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net,self).__init__()
        self.rnn = torch.nn.RNN(input_size,hidden_size,batch_first = True)
        self.fc = torch.nn.Linear(hidden_size,output_size,bias = True)

    def forward(self,x):
        x,_status = self.rnn(x)
        x = self.fc(x)
        return x   

#신경망 정의 생성자 같은경우에는 입력사이즈 은닉사이즈 출력사이즈 3개를 가지는 인자로 한다.
#Rnn 함수를 이용하여 매개변수로 받은 값이 입력과 은닉이 되고 전결합층이 은닉과 출력을 담당하게 된다.

net = Net(input_size,hiddensize,outputsize)
#연결시키기

outputs = net(X)
print(outputs.shape)

print(outputs.view(-1, input_size).shape)
#3차원 벡터를 2차원으로 줄이기
print(Y.view(-1).shape)
#2차원 벡터를 1차원으로 줄이기
criterion = nn.CrossEntropyLoss()
#다중클래스분류인 소프트맥스회귀함수를 손실함수로 정의
optimizer = optim.Adam(net.parameters(), lr)


for i in range(100):
    
    optimizer.zero_grad()
    outputs = net(X)
    #순전파
    loss = criterion(outputs.view(-1, input_size), Y.view(-1)) # view를 하는 이유는 Batch 차원 제거를 위해
    loss.backward() # 기울기 계산
    optimizer.step() # 아까 optimizer 선언 시 넣어둔 파라미터 업데이트

    # 아래 세 줄은 모델이 실제 어떻게 예측했는지를 확인하기 위한 코드.
    result = outputs.data.numpy().argmax(axis=2) # 최종 예측값인 각 time-step 별 5차원 벡터에 대해서 가장 높은 값의 인덱스를 선택
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)