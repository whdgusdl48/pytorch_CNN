import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
import random

#세팅 부분
random.seed(5)
torch.manual_seed(5)
#각각의 시드값을 준다.

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 10
#배치 사이즈는 64 학습률은 0.001퍼 반복횟수는 10회 하이퍼파라미터로 설정해준다.

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

#디바이스 설정 쿠다일지 cpu인지 쿠다가 사용가능하면 쿠다사용하게 한다.

Text = data.Field(sequential=True, batch_first=True, lower=True)
Label = data.Field(sequential=True, batch_first=True)

#Text필드와 라벨 필드를 정의한다 순차적으로 처음 배치를 하게 하고 대문자를 전부 소문자로 바꿔준다
#데이터를 로드 한다.

trainset, testset = datasets.IMDB.splits(Text, Label)

#IMDB에 있는 데이터를 TEXT,LABEL로 분할한다. 

print('trainset의 구성 요소 출력 : ', trainset.fields)


#단어 집합을 만든다. 
Text.build_vocab(trainset, min_freq = 5)
Label.build_vocab(trainset)

#Text는 최소 5번이상 등장한 단어만을 단어 집합에 추가한다는 의미를 가지고 있음
vocab_size = len(Text.vocab)
n_classes = 2
print('단어 집합의 크기 : {}'.format(vocab_size))
print('클래스의 개수 : {}'.format(n_classes))

#데이터 로더 생성
trainset, valset = trainset.split(split_ratio=0.8)
#훈련 데이터를 8대2로 생성해준다.
train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)
#버켓이터레이터를 통해서 각각 샘플과 레이블이 64개 단위 묶음으로 저장

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iter)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iter)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(val_iter)))

batch = next(iter(train_iter)) # 첫번째 미니배치
print(batch.text.shape)

batch = next(iter(train_iter)) # 두번째 미니배치
print(batch.text.shape)

train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (trainset, valset, testset), batch_size=BATCH_SIZE,
        shuffle=True, repeat=False)

class GRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(n_vocab, embed_dim)
        #임베딩 층 생성
        self.dropout = nn.Dropout(dropout_p)
        #마지막 남는 배치를 다버린다. 
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)
        #GRU를 선언하되 임베딩 층,매개변수로 입력받은 층의 개수                  
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size=x.size(0)) # 첫번째 히든 스테이트를 0벡터로 초기화
        x, _ = self.gru(x, h_0)  # GRU의 리턴값은 (배치 크기, 시퀀스 길이, 은닉 상태의 크기)
        h_t = x[:,-1,:] # (배치 크기, 은닉 상태의 크기)의 텐서로 크기가 변경됨. 즉, 마지막 time-step의 은닉 상태만 가져온다.
        self.dropout(h_t)
        logit = self.out(h_t)  # (배치 크기, 은닉 상태의 크기) -> (배치 크기, 출력층의 크기)
        return logit

    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = GRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    """evaluate model"""
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1) # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

best_val_loss = None
for e in range(1, EPOCHS+1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))

    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))        