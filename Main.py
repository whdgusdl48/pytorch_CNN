from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # 반응형 모드

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 4
img_name = landmarks_frame.iloc[n, 0] #csv에 있는 파일중 이미지 이름을 가지고 온다.
landmarks = landmarks_frame.iloc[n, 1:].as_matrix() # csv에 있는 파일중 이름 값을 제외한 나머지 좌표들을 배열형태로 저장한다.
landmarks = landmarks.astype('float').reshape(-1, 4) #랜드마크에 있는 배열들을 부동소수화 하여 각 행과 열에 맞게 재배치
# 즉 만약에 데이터가 136개가 할당되어있다면 위에처럼 4로 할당했을 때 사진에서 이미지 보여줄 때 136/4 즉 34개의 점으로 이미지를 분석한다.

print('Image name: {}'.format(img_name)) #이미지를 csv에 있는 목록별로 가지고 온다 즉 n의 값에 따라 사진이 변경됨
print('Landmarks shape: {}'.format(landmarks.shape)) #랜드마크의 매트릭스의 2차원 배열을 튜플형태로 나타낸 것
print('First 4 Landmarks: {}'.format(landmarks[:4])) # 첫 랜드마크 4개의 값을 출력한다.


def show_landmarks(image, landmarks): #이미지를 보여주는 함수 여기에 이미지는 이제 img_name이 될것이다.
    """Show image with landmarks"""
    """ 랜드마크(landmark)와 이미지를 보여줍니다. """
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r') #랜드마크의 이미지의 나눈 값만큼 흩뿌리기 함수로 체킹한다 여기서
                                                                           #여기서 s는 점의 크기 마커는 점의 형태 c는 색을 의미한다.
    plt.pause(0.001)  # 갱신이 되도록 잠시 멈춥니다.

plt.figure()
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), #여기서 show_landmark 함수를 통해서 이미지 경로와 csv에 있는 랜드마크를 나타내준다.
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
            idx = idx.tolist() #리스트형태로 변환해준다.

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0]) #파일경로를 매개변수로 설정하고 각 csv파일의 0번째 열의 값을 다가져온다.
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2) #여기까지 위의 참고하면 이해할 수 있다.
        sample = {'image': image, 'landmarks': landmarks} #기본적으로 사전형태로 값을 참조하는 방식

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/') #여기서 데이터셋 클래스의 매개변수를 현재 data폴더의 face폴더로 경로를 설정하고 csv파일 위치도 설정

print(len(face_dataset)) # 이미지 갯수만큼 데이터 셋이 생성됨을 확인할 수 있음
fig = plt.figure()
print(fig) #기본값 640*480형태로 나타내줌 값을 설정할수도 있음 출력결과 Figure(640x480)
for i in range(len(face_dataset)): # 이미지 갯수만큼 반복문을 돌립니다.
     sample = face_dataset[i]
     # print(i, sample['image'].shape, sample['landmarks'].shape)
     #샘플들의 이미지를 출력할것이다 랜드마크는 2차원 배열의 튜플형태로 나타나 질거고 이미지는 3차원형태의 튜플로 나타나질것이다.


     ax = plt.subplot(1, 4, i + 1) #여기까지 i값을 받아서 4까지 4개의 랜덤 값을 추출해준다고 생각하면 편합니다!
     plt.tight_layout() #자동으로 레이아웃을 설정해주는값 기본 값 1.08

     ax.set_title('Sample #{}'.format(i))
     ax.axis('off')
     show_landmarks(**sample) #이미지 4개를 보여준다 여기서 서브플랏의 값을 조정하면 이미지의 개수가 변경

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
        image, landmarks = sample['image'], sample['landmarks']

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

        return {'image': img, 'landmarks': landmarks} #리턴값은 객체체

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
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

scale = Rescale(256) #256크기로 자른다
crop = RandomCrop(128) #128의 크기로 랜덤구역으로 자른다.
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)]) #2개의 함수를 합친다. Compose함수를 통해 2개의 객체를 동시에 할수 있게 해준다.

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[2]
for i, tsfrm in enumerate([scale, crop, composed]): #반복문을 돌린다 열거형을 통해 256으로 크기를 줄인것 무작위로 자른것 두개를 동시에 합친것 이 3개를 사진에 구현한다.
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

#데이터셋을 통한 반복작업 가중치가 변경되기때문에 반복작업을 실행해주어야됨.
