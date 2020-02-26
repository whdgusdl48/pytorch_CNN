import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sentence = "Repeat is the best medicine for memory".split()
#공백을 기준으로 나눠준다.
voca = list(set(sentence))
#인덱스의 값이 겹치는게 있는지 판단하기위해 set함수 적용시켜서 리스트형태
word2Index = {tkn : i for i,tkn in enumerate(voca,1)}
word2Index['<unk>']=0
#각 단어의 고유정수를 할당하고 모르는 단어는 0으로 설정한다.
Index2Word = {v : k for k,v in word2Index.items()}
#인덱스를 통해 단어를 찾게 설정해주는 변수.

def build_data(sentence,word2Index):
    encoded = [word2Index[token] for token in sentence]#각 문자를 정수로 변환
    input_seq, label_seq = encoded[:-1], encoded[1:] #입력 시퀀스와 레이블 시퀀스를 분리 
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) #각각의 배치 차원 추가
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) #각각의 배치 차원 추가
    return input_seq, label_seq

X, Y = build_data(sentence,word2Index)
print(X,Y)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, # 워드 임베딩
                                            embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, # 입력 차원, 은닉 상태의 크기 정의
                                batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size) # 출력은 원-핫 벡터의 크기를 가져야함. 또는 단어 집합의 크기만큼 가져야함.

    def forward(self, x):
        # 1. 임베딩 층
        # 크기변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)
        # 2. RNN 층
        # 크기변화: (배치 크기, 시퀀스 길이, 임베딩 차원)
        # => output (배치 크기, 시퀀스 길이, 은닉층 크기), hidden (1, 배치 크기, 은닉층 크기)
        output, hidden = self.rnn_layer(output)
        # 3. 최종 출력층
        # 크기변화: (배치 크기, 시퀀스 길이, 은닉층 크기) => (배치 크기, 시퀀스 길이, 단어장 크기)
        output = self.linear(output)
        # 4. view를 통해서 배치 차원 제거
        # 크기변화: (배치 크기, 시퀀스 길이, 단어장 크기) => (배치 크기*시퀀스 길이, 단어장 크기)
        return output.view(-1, output.size(2))

vocab_size = len(word2Index)  # 단어장의 크기는 임베딩 층, 최종 출력층에 사용된다. <unk> 토큰을 크기에 포함한다.
input_size = 5  # 임베딩 된 차원의 크기 및 RNN 층 입력 차원의 크기
hidden_size = 20  # RNN의 은닉층 크기        

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
# 손실함수 정의
loss_function = nn.CrossEntropyLoss() # 소프트맥스 함수 포함이며 실제값은 원-핫 인코딩 안 해도 됨.
# 옵티마이저 정의
optimizer = optim.Adam(params=model.parameters())
decode = lambda y: [Index2Word.get(x) for x in y]

for step in range(201):
    # 경사 초기화
    optimizer.zero_grad()
    # 순방향 전파
    output = model(X)
    # 손실값 계산
    loss = loss_function(output, Y.view(-1))
    # 역방향 전파
    loss.backward()
    # 매개변수 업데이트
    optimizer.step()
    # 기록
    if step % 40 == 0:
        print("[{:02d}/201] {:.4f} ".format(step+1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        print(" ".join(["Repeat"] + decode(pred)))
        print()