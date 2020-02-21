import urllib.request
import pandas as pd
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.data import Iterator

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
print(df.head())
# read_csv파일을 통해 파일을 읽어온다.
train_df = df[:25000]
#훈련할 데이터는 25000개
test_df = df[25000:]
#테스트할 데이터도 25000개
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20
                  )
LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True
                   )

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
print(vars(train_data[0]))

TEXT.build_vocab(train_data, min_freq = 10 , max_size = 10000)
print(len(TEXT.vocab))

batch_size = 5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)

batch = next(iter(train_loader))
print(batch.text)