import torch
import torch.nn.functional as F

torch.manual_seed(1)

z1= torch.rand(3,5 , requires_grad= True)
# 3*5 행렬을 구현
z = torch.FloatTensor([1,2,3])
# 1*3행렬을 구현
print(z)
hypothesis = F.softmax(z1, dim=1)
#softMax 함수를 통해 가설을 세운다. dim=1? 왜냐하면 2차원행렬을 1차원으로 만들기 위해서 dim
print(hypothesis)

y = torch.randint(5, (3,)).long()
#torch.randint => 5까지 3개의 인덱스를 랜덤으로 넣는다.
print(y)

y_one_hot = torch.zeros_like(hypothesis)
# 3*5행렬을 모두 0으로 만들고
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
# 가설에 y의 unsqueeze(1)을 통해 1*3을 3*1로 바꾼다. 그 안에 인덱스의 값위치에 1을 넣어준다.
print(y_one_hot)

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

print(cost)

#high level

print(torch.log(F.softmax(z1 , dim=1)))
print(F.log_softmax(z1, dim=1))
print(F.cross_entropy(z1, y))