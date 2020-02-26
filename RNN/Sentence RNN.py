import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#데이터 전처리 과정
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

#문장선언
char_set = list(set(sentence))
#중복되는거 제외
char_dic = {c : i for i,c in enumerate(char_set)}
#사전형태로 각각의 문자열을 정수형태의 고유번호를 할당시켜준다.

hidden_size = 25
# 입력층의 길이가 25여서 은닉층도 25
sequence_length = 10 #임의 숫자를 지정하여 끊어서 샘플을 만들겁니다.
lr = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    x_data.append([char_dic[c] for c in x_str])  # x str to index
    y_data.append([char_dic[c] for c in y_str])  # y str to index

x_one_hot = [np.eye(25)[x] for x in x_data] # x 데이터는 원-핫 인코딩
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))


#모델 구현하기

class Net(torch.nn.Module):

    def __init__(self,input_dim,hidden_dim,layers):
        super(Net,self).__init__()
        self.rnn = nn.RNN(input_dim,hidden_dim,num_layers=layers, batch_first= True)
        self.fc = nn.Linear(hidden_dim,hidden_dim, bias=True)

    def forward(self,x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x    
#클래스에서 층을 2개로 늘린다. 입력은 25 은닉층 25의 2개이 층으로 한다. 
net = Net(len(char_dic),hidden_size,2)
#170,10,25의 3차원의 텐서형태가 된다.
criterion = nn.CrossEntropyLoss()
#소프트맥스 회귀함수로 손실함수로 정의한다.
optimizer = optim.Adam(net.parameters(), lr)

for i in range(150):
    optimizer.zero_grad()
    #미분시작
    outputs = net(X) # (170, 10, 25) 크기를 가진 텐서를 매 에포크마다 모델의 입력으로 사용
    #순전파
    loss = criterion(outputs.view(-1, 25), Y.view(-1))
    #output 170 10 25텐서에서 2차원으로 바꾼다. 170,10으로 Y는 170,10 에서 1700의 크기를 가지게됨
    loss.backward()
    #역전파
    optimizer.step()
    #값 저장

    # results의 텐서 크기는 (170, 10)
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0: # 처음에는 예측 결과를 전부 가져오지만
            predict_str += ''.join([char_set[t] for t in result])
        else: # 그 다음에는 마지막 글자만 반복 추가
            predict_str += char_set[result[-1]]

    print(predict_str)

