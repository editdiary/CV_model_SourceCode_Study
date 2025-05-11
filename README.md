# Computer Vision Model Source Code Study

본 Repository는 컴퓨터 비전 분야의 대표적인 오픈소스 모델인 Detectron2와 YOLO의 구현 코드를 분석하고 학습하기 위한 저장소입니다.

## 프로젝트 목적
- 딥러닝 모델의 코드 구현을 바닥부터 이해하고 학습
- Detectron2와 YOLO의 아키텍처 및 구현 방식 분석
- 실제 프로덕션 수준의 코드 구조와 설계 패턴 학습

## 파일 구조
```
CV_model_SourceCode_Study/
├── README.md
├── study_src/          # 학습한 내용을 정리하고 실습하는 코드
│   └── your_code.py
└── external/           # 분석 대상이 되는 외부 라이브러리
    ├── yolov12/        # YOLO 구현 코드
    └── detectron2/     # Detectron2 구현 코드
```

## External Dependencies
이 프로젝트는 다음 오픈소스 프로젝트들의 코드를 학습 목적으로 사용합니다:

- [Ultralytics YOLOv12](https://github.com/sunsmarterjie/yolov12)
  - YOLO(You Only Look Once) 모델의 구현 코드
  - 객체 검출을 위한 실시간 딥러닝 모델

- [Detectron2](https://github.com/facebookresearch/detectron2)
  - Facebook AI Research에서 개발한 객체 검출 프레임워크
  - Mask R-CNN, Faster R-CNN 등 다양한 모델 구현 포함

## 학습 내용
- [ ] YOLO 아키텍처 분석
- [ ] Detectron2 프레임워크 구조 분석
- [ ] 모델 구현 패턴 학습
- [ ] 실제 프로덕션 코드 구조 이해

## 참고사항
본 repo는 학습 목적으로 외부 라이브러리의 코드를 분석하고 있습니다.<br>
각 라이브러리의 라이센스를 준수하며, 원본 코드의 출처를 명시하고 있습니다.