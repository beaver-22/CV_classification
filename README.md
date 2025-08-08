# ResNet vs ViT: CutMix/MixUp

!!!!data loader는 직접 구현해야 한다!!!!

## 📋 목차

- [연구 개요](#-연구-개요)
- [핵심 가설](#-핵심-가설)
- [실험 설계](#-실험-설계)
- [평가 지표](#-평가-지표)
- [설치 및 실행](#-설치-및-실행)
- [실험 결과](#-실험-결과)
- [시각화](#-시각화)
- [분석 도구](#-분석-도구)
- [논문 기반 이론적 근거](#-논문-기반-이론적-근거)


## 🎯 연구 개요

**Vision** **Transformer** **(ViT)**와 **ResNet**에서 **CutMix**와 **MixUp** augmentation 기법 적용시 효과를 체계적으로 분석합니다. 두 아키텍처는 본질적으로 전역적 특징 학습(Transformer) vs 지역적 특징 학습(CNN)이라는 점에서 학습 방식과 결과에서 뚜렷한 차이를 보입니다. 이 때 사진을 조각내 연결하는 **CutMix**와 **MixUp** 이라는 두 가지 다른 augmentation 방식이 두 모델 학습에 있어 어떤 영향을 미치는 지에 관해서 탐구합니다.

<img width="1400" height="346" alt="image" src="https://github.com/user-attachments/assets/eabebd2d-7913-4de1-996d-4513c858dfa3" />


### 🔍 핵심 연구 질문
- ViT와 ResNet은 정말 CutMix에서 더 큰 성능 향상을 보이는가? 어느 정도의 성능 향상을 보이는가?
- ViT와 ResNet은 정말 Mixup에서 더 큰 성능 향상을 보이는가? 어느 정도의 성능 향상을 보이는가?
- Training with pretrained model weight vs Training from scratch 여부가 위의 성능 향상/하락에 어떤 영향을 미치는가?


## 🎯 핵심 가설

### Main Hypothesis
> **"ViT는 ResNet 대비 MixUp에서 더 큰 성능 향상을, CutMix에서는 상대적으로 작은 성능 향상을 보일 것이다"**

### 세부 가설
| 가설 | 내용 | 이론적 근거 |
|------|------|-------------|
| **H1** | ResNet의 CNN 구조는 CutMix의 spatial mixing에서 더 큰 이익을 얻을 것 | CNN의 지역적 특징 학습 능력! |
| **H2** | ViT의 Transformer 구조는 MixUp의 feature-level mixing에서 더 나은 성능을 보일 것 | ViT의 전역적 특징 학습 능력! |
| **H3** | Pretrained model 여부는 위의 두 **H1**, **H2** 세부가설에 영향을 미치지 않을 것 | Orthogonal한 조건임 |


## 📊 실험 설계

### 1. CIFAR100 dataset 실험

| 실험군 | 모델 | Augmentation |  Pretrained | 목적 |
|--------|------|--------------|---------|------------|
| **Control-1** | ResNet50 | Default |  ✗ | 베이스라인 설정 |
| **Exp-1** | ResNet50 | MixUp |  ✗ | CNN-MixUp 효과 |
| **Exp-2** | ResNet50 | CutMix |  ✗ | CNN-CutMix 효과 |
| **Control-2** | ViT-Small/16 | Default |  ✗ | 베이스라인 설정 |
| **Exp-3** | ViT-Small/16 | MixUp |  ✗ | ViT-MixUp 베이스라인 |
| **Exp-4** | ViT-Small/16 | CutMix |  ✗ | ViT-CutMix 효과 |


### 2. Tiny-ImageNet (200) 실험

| 실험군 | 모델 | Augmentation |  Pretrained | 목적 |
|--------|------|--------------|---------|------------|
| **Control-3** | ResNet50 | Default |  ✗ | 베이스라인 설정 |
| **Exp-5** | ResNet50 | MixUp |  ✗ | CNN-MixUp 효과 |
| **Exp-6** | ResNet50 | CutMix |  ✗ | CNN-CutMix 효과 |
| **Control-4** | ViT-Small/16 | Default |  ✗ | 베이스라인 설정 |
| **Exp-7** | ViT-Small/16 | MixUp |  ✗ | ViT-MixUp 베이스라인 |
| **Exp-8** | ViT-Small/16 | CutMix |  ✗ | ViT-CutMix 효과 |


### 3. Pretrained Model 실험
Pretrained weight 여부가 위 실험에 영향을 미치지 않는지 검증하기 위한 목적

| 실험군 | 데이터셋 | 모델 | Augmentation |  Pretrained | 
|--------|------|------|--------------|---------|
| **Control-5** | CIFAR100 | ResNet50 | Default |  ✓ | 
| **Exp-9** | CIFAR100 | ResNet50 | MixUp |  ✓ | 
| **Exp-10** | CIFAR100 | ResNet50 | CutMix |  ✓ | 
| **Control-6** | CIFAR100 | ViT-Small/16 | Default |  ✓ | 
| **Exp-11** | CIFAR100 | ViT-Small/16 | MixUp |  ✓ | 
| **Exp-12** | CIFAR100 | ViT-Small/16 | CutMix |  ✓ | 
| **Control-7** | Tiny-ImageNet | ResNet50 | Default |  ✓ | 
| **Exp-13** | Tiny-ImageNet | ResNet50 | MixUp |  ✓ | 
| **Exp-14** | Tiny-ImageNet | ResNet50 | CutMix |  ✓ | 
| **Control-8** | Tiny-ImageNet | ViT-Small/16 | Default |  ✓ | 
| **Exp-15** | Tiny-ImageNet | ViT-Small/16 | MixUp |  ✓ | 
| **Exp-16** | Tiny-ImageNet | ViT-Small/16 | CutMix |  ✓ | 

### 실험 설계
```python
# 실험 config
experiment_configs = {
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': 0.001,
    'optimizer': 'Adam'
}
```

## 📈 평가 지표

### 1. 기본 성능 지표 (Primary Metrics)

```python
primary_metrics = {
    'accuracy': {
        'top1': '가장 높은 확률 예측의 정확도',
        'top5': '상위 5개 예측 중 정답 포함 비율',
    },
    'precision_recall': {
        'precision': '예측된 클래스 중 실제 정답 비율(클래스 평로 계산 후 평균)',
        'recall': '실제 클래스 중 예측된 정답 비율(클래스 별로 계산 후 평균)',
        'f1-score': '각 클래스 F1의 평균',
    }
}
```

## 🏋️‍♀️ 학습 진행
<img width="3370" height="2624" alt="image" src="https://github.com/user-attachments/assets/6f164a13-c903-46f0-9c57-01e5ecf0aa92" />


## 🛠 설치 및 실행

### 환경 설정

```bash
# 저장소 클론
git clone https://github.com/beaver-22/CV_classification.git
cd CV_classification

# 필수 패키지 설치
pip install -r requirements.txt
```

### 종합 실험 실행
전체 실험 파이프라인 병렬 실행(Pretrained model 실험 포함됨)

#### CIFAR100 실험
```bash run_cifar.sh```

#### Tiny-ImageNet (200) 실험
```bash run_tiny.sh```

#### 전체 모델 평가
```eval.sh```

## 📊 실험 결과

### 전체성능 비교 요약
| 실험군 | 데이터셋 | 모델 | Augmentation | Pretrained | Top-1 Acc | Top-5 Acc | Recall | Precision | F1-score |
|--------|------|------|--------------|---------|-----------|--------|-----------|-------------|-------------|
| **Control-1** | CIFAR100 | ResNet50 | Default |  ✗ | 45.88% | 72.75% | 45.88% | 45.97% | 45.79% |
| **Exp-1** | CIFAR100 | ResNet50 | MixUp |  ✗ | 54.61%% | 79.11% | 54.61% | 54.60% | 53.94% |
| **Exp-2** | CIFAR100 | ResNet50 | CutMix |  ✗ | 57.50% | 83.15% | 57.50% | 57.68% | 57.09% |
| **Control-2** | CIFAR100 | ViT-Small/16 | Default |  ✗ | 23.83% | 49.19% | 23.83% | 23.84% | 23.08% |
| **Exp-3** | CIFAR100 | ViT-Small/16 | MixUp |  ✗ | 27.73% | 54.89% | 27.73% | 26.22% | 26.25% |
| **Exp-4** | CIFAR100 | ViT-Small/16 | CutMix |  ✗ |  18.63% | 44.09% | 18.63%  | 17.19% | 16.02% |
| **Control-5** | CIFAR100 | ResNet50 | Default |  ✓ | 63.67% | 86.94% | 63.67% | 63.70% | 63.52% |
| **Exp-9** | CIFAR100 | ResNet50 | MixUp |  ✓ | 68.64% | 88.36% | 68.64% | 68.60% | 68.32% |
| **Exp-10** | CIFAR100 | ResNet50 | CutMix |  ✓ | 70.45% | 90.49% | 70.45% | 70.37% | 70.17% |
| **Control-6** | CIFAR100 | ViT-Small/16 | Default |  ✓ | 43.79% | 70.47% | 43.79% | 43.69% | 43.52% |
| **Exp-11** | CIFAR100 | ViT-Small/16 | MixUp |  ✓ | 52.78% | 75.19% | 52.28% | 52.12% | 51.54% |
| **Exp-12** | CIFAR100 | ViT-Small/16 | CutMix |  ✓ | 56.48% | 81.74% | 56.48% | 56.12% | 55.96% |
| **Control-3** | Tiny-ImageNet | ResNet50 |  ✗ | Default | 36.34% | 61.53% | 36.34% | 36.14% | 36.05% |
| **Exp-5** | Tiny-ImageNet | ResNet50 | MixUp |  ✗ |  6.35% | 19.96% | 6.35% | 9.05% | 4.85% |
| **Exp-6** | Tiny-ImageNet | ResNet50 | CutMix |  ✗ | 29.23% | 55.38% | 29.23% | 32.49% | 28.25% |
| **Control-4** | Tiny-ImageNet | ViT-Small/16 | Default | ✗ | 9.96% | 25.03% | 9.96% | 9.03% | 8.10% |
| **Exp-7** | Tiny-ImageNet | ViT-Small/16 | MixUp |  ✗ |  12.78% | 31.28% | 12.78% | 10.74% | 10.68% |
| **Exp-8** | Tiny-ImageNet | ViT-Small/16 | CutMix |  ✗ |  15.60% | 36.80% | 15.60% | 13.30% | 13.06% |
| **Control-7** | Tiny-ImageNet | ResNet50 | Default |  ✓ | 66.58% | 84.79% | 66.58% | 66.91% | 66.52% |
| **Exp-13** | Tiny-ImageNet | ResNet50 | MixUp |  ✓ | 14.25% | 32.79% | 14.25% | 34.43% | 13.74% |
| **Exp-14** | Tiny-ImageNet | ResNet50 | CutMix |  ✓ | 48.77% | 74.36% | 48.77% | 52.71% | 48.27% |
| **Control-8** | Tiny-ImageNet | ViT-Small/16 | Default |  ✓ | 40.41% | 65.01% | 40.41% | 40.00% | 39.96% |
| **Exp-15** | Tiny-ImageNet | ViT-Small/16 | MixUp |  ✓ | 43.88% | 66.18% | 43.88% | 43.56% | 43.14% |
| **Exp-16** | Tiny-ImageNet | ViT-Small/16 | CutMix |  ✓ | 46.05% | 69.63% | 46.05% | 45.70% | 45.29% |


#### CIFAR100 실험
| 실험군 | 데이터셋 | 모델 | Augmentation | Pretrained | Top-1 Acc | Top-5 Acc | Recall | Precision | F1-score |
|--------|------|------|--------------|---------|-----------|--------|-----------|-------------|-------------|
| **Control-1** | CIFAR100 | ResNet50 | Default |  ✗ | 45.88% | 72.75% | 45.88% | 45.97% | 45.79% |
| **Exp-1** | CIFAR100 | ResNet50 | MixUp |  ✗ | 54.61%% | 79.11% | 54.61% | 54.60% | 53.94% |
| **Exp-2** | CIFAR100 | ResNet50 | CutMix |  ✗ | 57.50% | 83.15% | 57.50% | 57.68% | 57.09% |
| **Control-2** | CIFAR100 | ViT-Small/16 | Default |  ✗ | 23.83% | 49.19% | 23.83% | 23.84% | 23.08% |
| **Exp-3** | CIFAR100 | ViT-Small/16 | MixUp |  ✗ | 27.73% | 54.89% | 27.73% | 26.22% | 26.25% |
| **Exp-4** | CIFAR100 | ViT-Small/16 | CutMix |  ✗ |  18.63% | 44.09% | 18.63%  | 17.19% | 16.02% |

   ① ResNet50  
   - CutMix (57.50%) > MixUp (54.61%) > Default (45.88%)
   - CutMix는 해상도가 낮고 클래스가 많은 CIFAR100에서 ResNet 구조에 효과적

   ② ViT-Small/16  
   - MixUp (27.73%) > CutMix (18.63%) > Default (23.83%)
   - 50epoch이 학습이 부족함. 데이터셋의 크기도 부족해, fair한 비교라고 하기 힘듦


#### Tiny-ImageNet (200) 실험
| 실험군 | 데이터셋 | 모델 | Augmentation | Pretrained | Top-1 Acc | Top-5 Acc | Recall | Precision | F1-score |
|--------|------|------|--------------|---------|-----------|--------|-----------|-------------|-------------|
| **Control-3** | Tiny-ImageNet | ResNet50 | Default |  ✗ | 36.34% | 61.53% | 36.34% | 36.14% | 36.05% |
| **Exp-5** | Tiny-ImageNet | ResNet50 | MixUp |  ✗ |  6.35% | 19.96% | 6.35% | 9.05% | 4.85% |
| **Exp-6** | Tiny-ImageNet | ResNet50 | CutMix |  ✗ | 29.23% | 55.38% | 29.23% | 32.49% | 28.25% |
| **Control-4** | Tiny-ImageNet | ViT-Small/16 | Default | ✗ | 9.96% | 25.03% | 9.96% | 9.03% | 8.10% |
| **Exp-7** | Tiny-ImageNet | ViT-Small/16 | MixUp |  ✗ |  12.78% | 31.28% | 12.78% | 10.74% | 10.68% |
| **Exp-8** | Tiny-ImageNet | ViT-Small/16 | CutMix |  ✗ |  15.60% | 36.80% | 15.60% | 13.30% | 13.06% |

    - ResNet과 ViT모두 너무 적은 데이터셋으로 과소적합이 일어나, 비교가 어려움


#### Pretrained 실험
| 실험군 | 데이터셋 | 모델 | Augmentation | Pretrained | Top-1 Acc | Top-5 Acc | Recall | Precision | F1-score |
|--------|------|------|--------------|---------|-----------|--------|-----------|-------------|-------------|
| **Control-5** | CIFAR100 | ResNet50 | Default |  ✓ | 63.67% | 86.94% | 63.67% | 63.70% | 63.52% |
| **Exp-9** | CIFAR100 | ResNet50 | MixUp |  ✓ | 68.64% | 88.36% | 68.64% | 68.60% | 68.32% |
| **Exp-10** | CIFAR100 | ResNet50 | CutMix |  ✓ | 70.45% | 90.49% | 70.45% | 70.37% | 70.17% |
| **Control-6** | CIFAR100 | ViT-Small/16 | Default |  ✓ | 43.79% | 70.47% | 43.79% | 43.69% | 43.52% |
| **Exp-11** | CIFAR100 | ViT-Small/16 | MixUp |  ✓ | 52.78% | 75.19% | 52.28% | 52.12% | 51.54% |
| **Exp-12** | CIFAR100 | ViT-Small/16 | CutMix |  ✓ | 56.48% | 81.74% | 56.48% | 56.12% | 55.96% |
| **Control-7** | Tiny-ImageNet | ResNet50 | Default |  ✓ | 66.58% | 84.79% | 66.58% | 66.91% | 66.52% |
| **Exp-13** | Tiny-ImageNet | ResNet50 | MixUp |  ✓ | 14.25% | 32.79% | 14.25% | 34.43% | 13.74% |
| **Exp-14** | Tiny-ImageNet | ResNet50 | CutMix |  ✓ | 48.77% | 74.36% | 48.77% | 52.71% | 48.27% |
| **Control-8** | Tiny-ImageNet | ViT-Small/16 | Default |  ✓ | 40.41% | 65.01% | 40.41% | 40.00% | 39.96% |
| **Exp-15** | Tiny-ImageNet | ViT-Small/16 | MixUp |  ✓ | 43.88% | 66.18% | 43.88% | 43.56% | 43.14% |
| **Exp-16** | Tiny-ImageNet | ViT-Small/16 | CutMix |  ✓ | 46.05% | 69.63% | 46.05% | 45.70% | 45.29% |

- 오히려 실험을 통해 데이터셋이 작은 상황에서는 Pretrained model을 통한 비교가 적합할 수 있다고 생각하게 됨
- 모델에 상관없이 CutMix 성능이 MixUp 성능보다 우월함을 확인 가능함
- 

## 📑 결과 해석
1. **가설검정**
   **"ViT는 ResNet 대비 MixUp에서 더 큰 성능 향상을, CutMix에서는 상대적으로 작은 성능 향상을 보일 것이다"**
   ➜ CIFAR100-Pretrained(X) 실험에서 해당 가설에 부합하는 결과를 보이기도 했지만, 사실상 과소적합으로 인해 해석의 어려움이 존재했다.
   ➜ Pretrained model을 통한 비교에서는 CutMix를 적용한 모델이 한결!같이 높은 성능을 보였다.(가설에 위배)
   > 따라서 기존의 가설은 참이라 할 수 없다!

3. **Pretrained Weight 효과**  
   - 사전학습 유무가 생각 이상으로 성능에 압도적으로 큰 영향을 미쳤다.  
     -  ResNet50-CIFAR100: +17.8 p(45.9 → 63.7%).  
     -  ViT-Small-CIFAR100: +19.9 p(23.8 → 43.8%).  
     -  Tiny-ImageNet에서도 모든 모델이 사전학습 시 평균 +25 p 이상 상승.  
   - 따라서 _“사전학습 + 적절한 증강”_ 조합이 기본 전략이다.
  
4. **기타**
   - Recall은 여러 class classification에서는 수식적으로 accuracy와 동일해진다 -> 의미 없음...
   - Precision, F1-score도 유사한 결과가 도출 된 것으로 보아, 대부분의 논문에서 accuracy를 중시하는 이유가 있었음!
   - 충분한 epoch으로 실험 못한 아쉬움...

### 2. Best 조합 (Top-1 기준)

| 데이터셋 | 최적 조합 | Top-1 / Top-5 |
|-----------|-----------|---------------|
| CIFAR100  | **ResNet50 + CutMix + Pretrained** | 70.45% / 90.49% |
| Tiny-ImageNet | **ResNet50 + Default + Pretrained** | 66.58% / 84.79% |
