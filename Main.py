from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
from skimage import io, transform,color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
import torchvision.transforms as transforms

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드

landmarks_frame = pd.read_csv('data/train_image/train_visual_reading.csv');

img_name = [];
landmarks = [];
classes = [];
for i in range(0,107):
    img_name.append(landmarks_frame.iloc[i, 0])
    landmarks.append(landmarks_frame.iloc[i, 2:].as_matrix())
    classes.append(landmarks_frame.iloc[i,1]);
    landmarks[i] = landmarks[i].astype('float').reshape(-1,2)

def show_landmarks(image, landmarks): #이미지를 보여주는 함수 여기에 이미지는 이제 img_name이 될것이다.
    plt.imshow(image)
    plt.axis("off")
    for i in range(len(landmarks)-1):
        if(classes[i] == 1):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='b') #랜드마크의 이미지의 나눈 값만큼 흩뿌리기 함수로 체킹한다 여기서
        elif(classes[i] == 2):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='r')
        elif (classes[i] == 3):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='black')
        elif (classes[i] == 4):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='g')
        elif (classes[i] == 5):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='pink')
        elif (classes[i] == 6):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='grey')
        elif (classes[i] == 7):
            plt.scatter(landmarks[i][:, 0], landmarks[i][:, 1], s=10, marker='.', c='purple')
    plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.
def show_landmarks2(classes,image, landmarks): #이미지를 보여주는 함수 여기에 이미지는 이제 img_name이 될것이다.
    plt.imshow(image)
    for i in range(len(landmarks)-1):
        if(classes == 1):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='b') #랜드마크의 이미지의 나눈 값만큼 흩뿌리기 함수로 체킹한다 여기서
        elif(classes == 2):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        elif (classes == 3):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='black')
        elif (classes == 4):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='g')
        elif (classes == 5):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='pink')
        elif (classes == 6):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='grey')
        elif (classes == 7):
            plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='purple')
    plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.

plt.figure()
show_landmarks(cv2.imread(os.path.join('data/train_image/', img_name[0])), #여기서 show_landmark 함수를 통해서 이미지 경로와 csv에 있는 랜드마크를 나타내준다.
               landmarks)
plt.show()

class FaceLandmarksDataset(Dataset): #DataSet 클래스 정의 여기서 데이터집단을 받아온다.
                                     #데이터셋을 상속하고 아래와 같이 오버라이딩을 해야한다.
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): #길이 반환
        return len(self.landmarks_frame)

    def __getitem__(self, idx): #이미지 반환 함수
        if torch.is_tensor(idx): #텐서형태의면
            print(idx)
            idx = idx.tolist() #리스트형태로 변환해준다.

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0]) #파일경로를 매개변수로 설정하고 각 csv파일의 0번째 열의 값을 다가져온다.
        image = cv2.imread(img_name)
        classes = self.landmarks_frame.iloc[idx,1]
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2) #여기까지 위의 참고하면 이해할 수 있다.
        sample = {'image': image, 'classes' : classes ,'landmarks': landmarks} #기본적으로 사전형태로 값을 참조하는 방식

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='data/train_image/train_visual_reading.csv',
                                    root_dir='data/train_image/') #여기서 데이터셋 클래스의 매개변수를 현재 data폴더의 face폴더로 경로를 설정하고 csv파일 위치도 설정

image1 = [];
for i in range(len(face_dataset)):
    if(face_dataset.landmarks_frame.iloc[i, 0] == "001.tif"):
        image1.append(face_dataset[i])
    else: break

fig = plt.figure()
for i in range(len(image1)): # 이미지 갯수만큼 반복문을 돌립니다.
     sample = image1[i]
     print(i, sample['image'].shape, sample['landmarks'].shape)
     #샘플들의 이미지를 출력할것이다 랜드마크는 2차원 배열의 튜플형태로 나타나 질거고 이미지는 3차원형태의 튜플로 나타나질것이다.

     ax = plt.subplot(1, 4, i + 1) #여기까지 i값을 받아서 4까지 4개의 랜덤 값을 추출해준다고 생각하면 편합니다!
     plt.tight_layout() #자동으로 레이아웃을 설정해주는값 기본 값 1.08

     ax.set_title('Sample #{}'.format(i))
     ax.axis('off')
     show_landmarks2(**sample)

     if i == 3: #만약 i가 3이면 이반복문을 빠져나오게 한다.
         plt.show()
         break

class Rescale(object): #주어진 크기의 샘플을 조정하는 클래스
    # Args:
    #     output_size(tuple or int) : 원하는 사이즈 값
    #         tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
    #         int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample): #샘플의 이미지값과 샘플의 랜드마크 값을  가져와서 변수의 저장
        image, landmarks, classes = sample['image'], sample['landmarks'],sample['classes']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w)) #이미지를 리사이징을 통해 새로운 값의 높이와 넓이를 변환해준다.

        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'classes': classes, 'landmarks': landmarks} #리턴값은 객체체

class RandomCrop(object):
        # output_size (tuple or int): 줄이고자 하는 크기입니다.
        #                 int라면, 정사각형으로 나올 것 입니다.
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2 #2값을 주어서 정사각형으로 만듬
            self.output_size = output_size

    def __call__(self, sample): # 호출 이미지와 랜드마크의 샘플을 가져와서 랜덤값을 통해 top,left를 통해 랜드마크에 적용된 값을 빼준다. 그러면 이미지에 맞게 각 랜드마크가 새겨질것.
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'classes': classes, 'landmarks': landmarks}


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, landmarks, classes = sample['image'], sample['landmarks'], sample['classes']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'classes' : classes,
                'landmarks': torch.from_numpy(landmarks)}

scale = Rescale(256) #256크기로 자른다
crop = RandomCrop(128) #128의 크기로 랜덤구역으로 자른다.
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)]) #2개의 함수를 합친다. Compose함수를 통해 2개의 객체를 동시에 할수 있게 해준다.

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = image1[2]
for i, tsfrm in enumerate([scale, crop, composed]): #반복문을 돌린다 열거형을 통해 256으로 크기를 줄인것 무작위로 자른것 두개를 동시에 합친것 이 3개를 사진에 구현한다.
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks2(**transformed_sample)

plt.show()

#데이터셋을 통한 반복작업 가중치가 변경되기때문에 반복작업을 실행해주어야됨.
transformed_dataset = FaceLandmarksDataset(csv_file='data/train_image/train_visual_reading.csv',
                                           root_dir='data/train_image/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(),sample['classes'], sample['landmarks'].size())

    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=1,shuffle=False)


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        # plt.axis('off')
        plt.ioff()
        plt.show()
        break

class Net(nn.Module) :

    def __init__(self):
        #상속은 기본으로 받기
        super(Net, self).__init__()

        #self.layer1 = nn.Sequential(
        #    nn.Conv2d(3, 6, 5),

        #    nn.ReLU(inplace=True),
        #    nn.MaxPool2d(2, 2),
        #)
        #self.layer2 = nn.Sequential(
        #    nn.Conv2d(6, 16, 5),
        #    nn.ReLU(),
        #    nn.MaxPool2d(2, 2)
        #)
        #self.fc = nn.Sequential(
        #    nn.Linear(16 * 5 * 5, 120),
        #    nn.Linear(120, 84),
        #    nn.Linear(84, 10)
        #)
        #nn.Conv2d의 인자는 input, output, pixel임
        self.conv1 = nn.Conv2d(3, 224, 5)
        #앞서 6개의 output이기 때문에 6을 input으로
        self.conv2 = nn.Conv2d(224, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        #conv2결과 16층, 5픽셀이므로 16*5*5
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x) :
        #relu 활성함수 사용, 결과가 정사각형이면 n, 아니면 n,m

        #x = self.to(self.device)
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.fc(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#net 객체 생성, print는 정보를 알기위해 tutorial에서 한거
net = Net()
print(net)

net = net.double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs = data['image']
        labels = data['classes']

        optimizer.zero_grad()
        print(inputs.shape)
        outputs = net(inputs)
        #outputs.to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(' [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

