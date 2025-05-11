import torch
import torch.nn as nn

# YOLOv1 아키텍처는 주로 Conv 레이어와 MaxPool 레이어로 구성됩니다.
# conv_block은 일반적인 Conv + Leaky ReLU 조합을 나타냅니다.
# 원 논문에는 BatchNorm이 없으므로 bias=False는 필수는 아니지만,
# 일부 구현체에서 Conv layer 뒤에 BatchNorm을 사용하기도 합니다. 여기서는 원본에 가깝게 bias=True를 기본으로 합니다.
def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
        nn.LeakyReLU(0.1, inplace=True) # YOLOv1은 Leaky ReLU (음수 기울기 0.1) 사용
    )

class YOLOv1(nn.Module):
    # split_size (S): 이미지를 나눌 그리드 크기 (e.g., 7)
    # num_boxes (B): 각 그리드 셀이 예측할 바운딩 박스 개수 (e.g., 2)
    # num_classes (C): 클래스 개수 (e.g., 20 for Pascal VOC)
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super(YOLOv1, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # YOLOv1의 컨볼루션 레이어들 (Darknet의 초기 구조와 유사)
        # 원 논문의 Table 1 또는 아키텍처 다이어그램을 기반으로 한 일반적인 구현 순서입니다.
        # 입력 이미지 크기를 448x448로 가정했을 때, 각 레이어를 통과하며 피처 맵 크기가 줄어듭니다.
        self.darknet = nn.Sequential(
            # Conv 1: 448x448x3 -> 224x224x64
            conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2), # 224x224x64 -> 112x112x64

            # Conv 2: 112x112x64 -> 112x112x192
            conv_block(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112x192 -> 56x56x192

            # Conv 3-5: 56x56x192 -> 56x56x512
            conv_block(192, 128, kernel_size=1, stride=1, padding=0),
            conv_block(128, 256, kernel_size=3, stride=1, padding=1),
            conv_block(256, 256, kernel_size=1, stride=1, padding=0),
            conv_block(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56x512 -> 28x28x512

            # Conv 6-14 (Repeat 4 times block: 512->256->512), then 512->1024
            # 최종적으로 28x28x512 -> 28x28x1024
            *([conv_block(512, 256, kernel_size=1, stride=1, padding=0),
               conv_block(256, 512, kernel_size=3, stride=1, padding=1)] * 4), # 8 Conv layers
            conv_block(512, 512, kernel_size=1, stride=1, padding=0),
            conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28x1024 -> 14x14x1024

            # Conv 15-20 (Final Conv layers)
            # 14x14x1024 -> 14x14x1024 (총 6개 Conv)
            conv_block(1024, 512, kernel_size=1, stride=1, padding=0),
            conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            conv_block(1024, 512, kernel_size=1, stride=1, padding=0),
            conv_block(512, 1024, kernel_size=3, stride=1, padding=1),
            conv_block(1024, 1024, kernel_size=3, stride=1, padding=1), # stride=1 유지

            # 마지막 stride=2 Conv 또는 풀링을 통해 7x7 피처맵 생성 (원 논문에는 Conv 20에서 stride 2)
            conv_block(1024, 1024, kernel_size=3, stride=2, padding=1), # 14x14x1024 -> 7x7x1024

            # 마지막 2개 Conv layers on 7x7 feature map => ## 이때의 7x7이 논문에서 말하는 grid를 의미
            ## 즉, grid를 나눈다는 건 최종 CNN layer가 7x7이 되는 거지 input image를 처음 나누고 시작하는 게 아니다.
            ## 즉, 넓은 receptive field를 가진 cnn layer의 최종 feature가 grid라고 표현한 의미인 거 같다.
            conv_block(1024, 1024, kernel_size=3, stride=1, padding=1),
            conv_block(1024, 1024, kernel_size=3, stride=1, padding=1),
        ) # 총 24개의 Conv 레이어

        # Fully Connected 레이어들
        # 마지막 Conv 레이어의 출력을 Flatten하여 FC 레이어의 입력으로 사용
        # 7x7x1024 피처맵 -> 49x1024 = 50176 크기의 벡터
        self.fc_layers = nn.Sequential(
            nn.Flatten(), # 피처 맵을 1차원 벡터로 펼침

            # 첫 번째 FC 레이어 (원 논문에서 4096 크기 사용)
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1, inplace=True),

            # 드롭아웃 레이어가 종종 추가되지만, 순수 아키텍처 설명에는 생략합니다.
            # nn.Dropout(p=0.5),

            # 두 번째 FC 레이어: 최종 출력을 S*S*(B*5+C) 크기로 만듬
            # 각 셀(S*S개)당 B개의 박스 예측 (x, y, w, h, confidence: 5개 값) + C개의 클래스 확률
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        # 입력 x 형태: (batch_size, in_channels, height, width) e.g., (N, 3, 448, 448)

        # 컨볼루션 레이어들을 통과
        x = self.darknet(x)
        # darknet 통과 후 x 형태: (batch_size, 1024, self.S, self.S) e.g., (N, 1024, 7, 7)

        # Fully Connected 레이어들을 통과
        x = self.fc_layers(x)
        # fc_layers 통과 후 x 형태 (Flatten 됨): (batch_size, self.S * self.S * (self.B * 5 + self.C)) e.g., (N, 7*7*(2*5+20)=1470)

        # 최종 출력을 그리드 형태로 재구성
        # (batch_size, S, S, B*5+C) 형태로 만듬 (PyTorch는 기본적으로 채널을 맨 앞에 두는 경향이 있지만,
        # 탐지 결과 해석을 위해 (Batch, Height, Width, Channels) 형태로 재구성하는 것이 일반적입니다)
        output = x.view(-1, self.S, self.S, (self.B * 5 + self.C))

        # output 형태: (batch_size, self.S, self.S, self.B * 5 + self.C) e.g., (N, 7, 7, 30)
        # 이 최종 출력 텐서는 각 그리드 셀(7x7)에 대해 B개의 바운딩 박스 정보(x, y, w, h, confidence)와 C개의 클래스 확률을 포함합니다.

        return output


# S=7, B=2, C=20 설정 (Pascal VOC 기준)
# split_size=7
# num_boxes=2
# num_classes=20
# input_size = 448 # YOLOv1의 표준 입력 크기

# model = YOLOv1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
# print(model) # 모델 레이어 구조 출력