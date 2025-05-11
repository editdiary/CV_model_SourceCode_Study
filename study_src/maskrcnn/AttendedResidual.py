import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d # Deformable Convolution을 위해 필요

# 1. 학습 가능한 스케일(Scalar) 클래스
# 논문에서 Transformer Attention 모듈의 출력에 곱해지는 learnable scalar.
# 사전 학습된 모델의 초기 동작을 방해하지 않기 위해 0으로 초기화됩니다.
class Scale(nn.Module):
    def __init__(self, init_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

# 2. 간소화된 Transformer Attention Module (E3: Key Content Only)
# 논문에서 "0010" 설정에 해당하며, E3(키 콘텐츠만) 항만 활성화된 경우를 모방합니다.
# 이는 주로 공간적(spatial) saliency를 학습하여 중요한 영역에 집중하는 방식으로 구현됩니다.
class TransformerAttentionE3Only(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # E3 항을 계산하는 부분: u_m^T V_m^C x_k (여기서는 Conv2d로 간소화)
        # x_k (입력 피처)를 받아 중요도 점수(saliency score)를 출력합니다.
        # 출력 채널은 1개로 하여 각 공간 위치별 중요도를 나타냅니다.
        self.saliency_scorer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False), # V_m^C (차원 축소)
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, bias=False) # u_m^T (점수 생성)
        )
        
        # W_m' (value projection)은 여기서는 입력 x 자체를 값으로 사용하거나,
        # 또는 1x1 컨볼루션을 사용하여 값을 추출하는 것으로 간주합니다.
        # 여기서는 x에 직접 적용하는 것으로 가정합니다.
        # output_proj는 최종 출력을 위해 필요할 수 있습니다.

        # 논문의 E3 항은 softmax로 정규화된 후 가중 합에 사용되지만,
        # 이미지 인식 태스크에서 "key content only"는 종종 공간적 게이팅(spatial gating)
        # (즉, 특정 영역을 강조하거나 약화시키는 스케일링)으로 구현됩니다.
        # 여기서는 Sigmoid를 사용하여 [0, 1] 범위의 공간 가중치 맵을 생성합니다.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()

        # E3 점수 계산: 각 공간 위치의 중요도
        saliency_scores = self.saliency_scorer(x) # (B, 1, H, W)

        # Sigmoid를 적용하여 가중치 맵 생성 (0~1 사이의 값)
        attention_map = self.sigmoid(saliency_scores) # (B, 1, H, W)

        # 입력 피처에 가중치 맵을 곱하여 "attention"을 적용합니다.
        # 이는 특정 영역의 피처를 강조하거나 약화시키는 효과를 줍니다.
        attended_features = x * attention_map
        
        return attended_features

# 3. Attended Residual Block
# ResNet Bottleneck Block을 기반으로 Deformable Convolution과
# Transformer Attention Module (E3 Only)을 통합합니다.
class AttendedResidualBlock(nn.Module):
    expansion = 4 # ResNet Bottleneck Block의 채널 확장 비율

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, deformable_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 Conv를 Deformable Convolution으로 대체 (논문 Fig 2(a) 참고)
        # offset_conv는 Deformable Conv의 샘플링 오프셋을 예측합니다.
        # groups는 채널 그룹을 나눌 때 사용됩니다 (DeformConv2d의 매개변수).
        self.offset_conv = nn.Conv2d(
            out_channels, 
            deformable_kernel_size * deformable_kernel_size * 2, # 2 for (x,y) offset
            kernel_size=deformable_kernel_size,
            stride=stride,
            padding=deformable_kernel_size // 2,
            bias=False
        )
        # DeformConv2d는 실제 변형 가능한 컨볼루션 연산을 수행합니다.
        self.conv2 = DeformConv2d(
            out_channels, out_channels, 
            kernel_size=deformable_kernel_size,
            stride=stride,
            padding=deformable_kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Transformer Attention Module (E3 Only) 삽입
        self.attention_module = TransformerAttentionE3Only(out_channels)
        # 논문에 따라 어텐션 모듈 출력에 곱해지는 learnable scalar
        self.attention_scale = Scale(init_value=0.0) 
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample # 잔여 연결을 위한 다운샘플링 레이어 (ResNet에서 사용)

    def forward(self, x):
        identity = x

        # Conv1 -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Deformable Conv2 (3x3 Conv 대신) -> BN -> ReLU
        # DeformConv2d는 오프셋을 입력으로 받습니다.
        offsets = self.offset_conv(out)
        out = self.conv2(out, offsets)
        out = self.bn2(out)
        out = self.relu(out)

        # Transformer Attention Module (E3 Only) 적용
        # 어텐션 모듈 주변에 잔여 연결이 있습니다.
        attention_out = self.attention_module(out)
        attention_out = self.attention_scale(attention_out) # learnable scalar 곱하기
        out = out + attention_out # 어텐션 모듈의 잔여 연결

        # Conv3 -> BN
        out = self.conv3(out)
        out = self.bn3(out)

        # 잔여 연결
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
    


# 간단한 테스트
if __name__ == "__main__":
    # 모델 초기화
    in_channels = 256
    out_channels = 64 # bottleneck block의 첫 1x1 conv 출력 채널 (확장 후 256이 됨)
    stride = 1
    
    # downsample은 ResNet에서 stride가 1이 아닐 때 또는 채널 수가 변경될 때 사용
    # 여기서는 간단한 테스트이므로 None으로 둡니다.
    # 실제 ResNet에서는 `nn.Sequential(nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels * expansion))` 형태가 됩니다.
    
    block = AttendedResidualBlock(in_channels, out_channels, stride=stride)
    print(block)

    # 더미 입력 생성 (Batch_size, Channels, Height, Width)
    dummy_input = torch.randn(1, in_channels, 56, 56) 

    # forward 패스
    output = block(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape) # (1, 256, 56, 56)

    # 스케일 값 확인 (초기값은 0이므로 초기에는 어텐션 모듈 출력이 0에 가까울 것임)
    print(f"Attention scale initial value: {block.attention_scale.scale.item()}")

    # 트랜스포머 어텐션 모듈의 E3 스코어러의 bias가 없으므로,
    # attention_scale이 0에 가까울 때 output이 identity와 매우 비슷해야 합니다.
    # print(f"Difference from identity (initial): {(output - dummy_input).abs().sum()}")
    # 위의 차이 검증은 좀 더 복잡한 환경 (ResNet의 초기화 방식 등)에서 유의미합니다.
    # 현재 구현에서는 input channel과 output channel이 다르므로 직접 비교는 어렵습니다.
    # Bottleneck 블록은 input_channels -> out_channels -> out_channels -> out_channels * expansion 입니다.
    # 따라서 dummy_input과 output의 shape는 동일할 수 있습니다.
    # (1, 256, 56, 56) -> AttendedResidualBlock(in_channels=256, out_channels=64)
    # Conv1: 256->64
    # Conv2: 64->64 (Deformable)
    # Attention: 64->64
    # Conv3: 64->256
    # 그래서 output_shape은 (1, 256, 56, 56)이 됩니다.