# Music_Tagging_AI

## Overview
이 프로젝트는 유튜브 영상 제목 및 채널명 텍스트에서 아티스트(ARTIST)와 원곡명(SONG)을 자동으로 추출하고 태깅하는 개체명 인식 모델을 개발하는 것을 목표로 합니다.
커스텀 데이터셋을 구축하고 Hugging Face transformers 라이브러리의 BERT 모델을 파인튜닝하여, 실제 서비스 환경에 준하는 F1-Score 92%의 고성능 모델을 구현하였습니다.

| 지표 | ARTIST | SONG | Overall(Weighted Avg.) |
|------|--------|------|------------------------|
| **Precision** | 91% | 89% | 90% |
| **Recall** | 96% | 94% | 95% |
| **f1-score** | 93% | 92% | 92% | 


## Technical Deep Dive
### 클래스 불균형에 대응하는 커스텀 손실 함수 구현
개체명 인식 작업의 특성상 대부분의 토큰은 "O" 태그를 가지므로, 데이터에 심각한 클래스 불균형이 존재했습니다.
 * **기법** : TensorFlow/Keras에서 masked_sparse_categorical_crossentropy라는 커스텀 손실 함수를 구현했습니다.
   <br />
   * **마스킹** : -100 레이블을 가지는 패딩 및 서브워드 토큰을 손실을 계산에서 제외했습니다.
   * **클래스 가중치** : 'O' 태그에 낮은 가중치를, 그 외 태그에 높은 가중치를 할당하여 모델이 개체명 예측에 더 집중하도록 유도했습니다. (최종 가중치 : [0.2, 2, 2, 2, 2])
 * **최종 최적화** : 학습률을 1e-5에서 2e-5으로 높이고 가중치를 미세 조정하여 오탐을 줄였고, 최종적으로 <b>F1-Score 92%</b> 를 달성하였습니다.

## Tech Stack
 * **Language** : Python 3.11
 * **Deep Learning Framework** : TensorFlow 2.15, Keras
 * **NLP Library** : Hugging Face transformers
 * **Evaluation** : seqeval
 * **Model** : bert-base-multilingual-cased
