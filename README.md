# 🏠 Seoul House Price ML Challenge

본 프로젝트는 2025년 7월 진행된 **서울시 아파트 실거래가 예측** 머신러닝 경진대회의 결과물입니다. 회귀(Regression) 문제 해결을 목표로, 데이터 엔지니어링부터 모델링, 협업까지 전 과정을 담았습니다.

## 🚀 핵심 요약 (TL;DR)

- **주제**: 서울시 아파트 실거래가 예측을 위한 머신러닝 회귀 모델 개발
- **팀**: `3X+Y` (AI·통계·컴퓨터공학 전공 4인)
- **주요 기술**: Feature Engineering, Time Series CV, Ensemble (Voting, Stacking), XGBoost, LightGBM, CatBoost
- **핵심 성과**: 최종 RMSE **46,950** (대회 5위), 체계적인 협업 프로세스 구축
- **핵심 교훈**: 데이터 품질의 중요성, 시계열 특성 고려의 필요성, 모델 설명 가능성의 가치

## Team

## 👨‍💻 팀: 3X+Y

> "한 줄의 코드, 한 뼘의 통찰. 정확한 예측 모델을 향하여!"
> 

‘3X+Y’는 팀원들의 MBTI 유형 분포가 ‘3:1’인 점에서 착안한 이름입니다. AI, 금융·통계, 컴퓨터공학 등 다양한 배경을 가진 팀원들이 모여 시너지를 만들었습니다.


| ![김선민](https://github.com/user-attachments/assets/263154ec-efa6-473a-bd55-3737fb5741bc) | <img src="https://avatars.githubusercontent.com/u/128503571?v=4](https://avatars.githubusercontent.com/u/128503571?v=4" width="200"> | <img src="https://github.com/user-attachments/assets/5c04a858-46ed-4043-9762-b7eaf7b1149a" width="200"> | ![장윤정](https://github.com/user-attachments/assets/af6a9ff5-56fc-4cd5-b61e-61269a24278d) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
| [김선민](https://github.com/nimnusmik) | [김장원](https://github.com/jkim1209) | [최현화](https://github.com/iejob) | [장윤정](https://github.com/yjjang06) |
| 팀 매니징 및 모델링 | 데이터 수집, 전처리, 모델링, 발표 | DE / FE / QA / ML, 깃헙 관리 | 모델링 |


## Overview

- 본 아파트 가격 예측 챌린지는 참가자들이 서울 아파트의 실제 거래 가격을 정확하고 일반화된 모델로 예측하는 것을 목표로 합니다.

미리 시장 동향을 예측함으로써, 모델은 현명한 부동산 의사결정을 돕고 공정하며 효율적인 거래를 촉진할 수 있습니다. 참가자들은 또한 데이터 과학 및 머신러닝 분야에서 실질적인 경험을 쌓고, 다양한 아파트 특성 간의 관계를 탐구할 수 있습니다.

저희 팀5조의 서울 부동산 가격 예측 프로젝트 플로우는 다음과 같이 진행했습니다.

```mermaid
graph TD;
    %% 1. 프로젝트 초기 설정 및 원천 데이터 획득
    A[팀 5조 결성 및 레포 생성] --> B[Raw Data 다운로드];

    %% 2. 데이터 전처리 및 기본 병합
    B --> C{결측치 처리 방법 논의}--> G{최종 컬럼 선택 및 데이터셋 완성};
    B --> D[지오코딩으로 X, Y 좌표 결측치 채우기];
    D --> D1[교통편 8개 컬럼 병합: 지하철, 버스 관련]--> G{최종 컬럼 선택 및 데이터셋 완성};
    

    %% 3. 추가 외부 데이터 통합
    B --> E[추가 외부 데이터 통합];
    E --> E1[금리 데이터 병합];
    E1 --> E2[인구수 데이터 병합: 총인구수, 성비 남여]--> G{최종 컬럼 선택 및 데이터셋 완성};

    %% 4. 피처 엔지니어링
    B--> F[피처 엔지니어링]
    F --> F1[날짜 피처 생성: 계약년월, 계약일자, 계약년도, 계약월];
    F1 --> F2[아파트명 길이 / 홈페이지 유무 피처 생성];
    F2 --> F3[연식 피처 생성: 계약년도 - 건축년도];
    F3 --> F4[브랜드 등급 피처 생성] --> G{최종 컬럼 선택 및 데이터셋 완성};

    %% 5. 최종 데이터셋 구성 및 모델링 준비
    G --> Z[모델링];
    Z --> H[하이퍼파라미터 조정];
    H --> Z[모델링];
    H --> I[보고서 작성 및 제출];
    I --> Q[발표];

```

## 🤝 협업 방식 및 도구

**열린 소통, 상호 존중, 책임감, 주도성**을 핵심 가치로 삼았습니다.

- **Communication**: 매일 2회 정기 미팅(오전/오후) 및 **Slack**을 통한 실시간 소통
- **Task Management**: **GitHub Issues** (약 70개) 및 **Projects**를 활용한 체계적 업무 관리
- **Documentation**: **Notion**을 중앙 허브로 사용하여 회의록, 아이디어, To-Do 리스트 기록
- **Version Control**: **GitHub**를 ‘단일 진실 소스(Central Source of Truth)’로 삼고, 기능별 브랜치와 PR 워크플로우를 철저히 준수
- **Data & Asset Sharing**: **Google Drive**로 데이터셋 및 발표 자료 공유

---

## Timeline

- 모델링 전(7/7–7/11)
    - 7/7–7/8: 주제 정의·데이터 구조 탐색
    - 7/9–7/11: 외부 데이터 수집·결측치 처리·피처 엔지니어링
    - 회의:
        - 매일 **10:10** 스탠드업 (당일 목표·이슈 공유)
        - 매일 **18:30** 진행 상황 점검
    
- 모델링 전환(7/12–7/13)
    - 7/12–7/13:
        - A조 Insight 정리 (타깃 관계 분석·문서화)
        - B조 Feat Eng 분업 (이상치 처리·스케일링·인코딩·간단 모델링)
    - 회의: 매일 **18:00** 모델링 준비 회의

- **모델링·최종 검증(7/14–7/17)**
    - **7/14–7/17**: 모델 학습·추가 피처링 병행·최종 RMSE 검증
    - **회의**: 매일 **18:00** 결과 공유 회의


### Dev Environments

```bash
.
├── data
│ ├── logs
│ │ ├── geocoding_logs
│ │ ├── price_prediction_logs
│ │ └── transportation-features_logs
│ ├── processed
│ │ ├── cleaned_data
│ │ ├── geocoding
│ │ ├── params
│ │ ├── price-prediction
│ │ ├── submissions
│ │ └── transportation-features
│ └── raw
├── docs
│ └── pdf
├── font
│ └── NanumFont
├── images
│ └── price_prediction_hyunhwa
├── model
│ └── price_prediction_hyunhwa
├── notebooks
│ ├── csv
│ ├── geocoding-missing-coords
│ ├── price-prediction
│ └── transportation-features
└── src
├── data
└── log
└── pycache
```

### Directory Description 

1. data: 프로젝트의 모든 데이터(csv) 관련 파일 보관소
    
    • logs: 지오코딩·모델 학습·교통 피처 생성 과정의 로그
    
    • processed: 클리닝·지오코딩·파라미터·최종 예측·제출 파일 등 가공 데이터
    
    • raw: 제공받은 원본 CSV 파일(bus_feature, loanrate, population, subway_feature, train/test.csv)
    

1. docs: 프로젝트 산출물 및 템플릿용 PPT 파일
2. font : NanumGothic 폰트 파일
3. images: 모델별 시각화 이미지(피처 중요도, SHAP, 학습 곡선 등)
4. model: 버전별 학습된 모델 객체(.pkl)
5. notebooks: 주피터 파일 저장
    
    • csv: CSV 비교·리사이즈·제출 포맷용 실험 노트북
    
    • geocoding-missing-coords: 좌표 결측치 탐색 및 지오코딩 노트북
    
    • transportation-features: 교통 관련 파생변수 생성 노트북
    
    • price-prediction: 1~9버전 모델링 실험 노트북
    
6. src: python 파일 저장
    
    • data: 데이터 다운로드·정제·피처 엔지니어링·모델링 스크립트
    
    • log: 로거 구현 및 캐시 파일(**pycache**)


## EDA
1. 결측치 현황 파악
  - 전체 52개 컬럼 중 41개 컬럼 결측치 존재
  - 이 중 37개 컬럼의 결측치 비율 70% 이상
  - 공백 등 의미없는 값으로 채워진 컬럼도 존재
2. 결측치 탐색 및 처리 방향
  - 최빈값, 0등으로 대체하는 방식 시뮬레이션 진행
  - 값 유무에 따라 0/1 또는 공백으로 대체하는 방식 검토
  - `좌표X`, `좌표Y` 결측치 보완을 위해 지오코딩 적용 검토
3. 외부 데이터 조사 및 통합
  - **주택담보대출금리**가 집값에 영향을 줄 수 있다고 생각하여 금리 데이터 추가
  - **인구 밀도**가 집값에 영향을 줄 수 있다고 생각하여 인구 데이터 추가
4. 모델 학습용 컬럼 확정
  - 위 결과를 바탕으로 학습에 사용할 컬럼 확정

## Feature engineering
1. 결측치 처리
  - 결측치가 많은 변수 중, 논리적·직관적으로 필요 없다고 판단되는 변수들 제거
  - 교통 관련 파생변수 생성을 위해 `좌표X`, `죄표Y` 결측치 지오코딩을 이용하여 보완
  - `아파트명`, `단지분류` 등의 결측치 빈문자열('')로 처리
  - `홈페이지`, `사용허가여부` 등의 결측치는 값 유/무 여부를 1/0으로 처리
3. 날짜 변수
  - `계약일자` -> `계약년도`, `계약월`로 분리
  - `건축년도`와 `계약년도`를 조합하여 `연식` 파생변수 생성
3. 지역 및 교통 관련 변수
  - `시군구` -> `자치구`, `법정동`으로 분리
  - `강남3구여부`(강남, 서초, 송파) 파생변수 생성
  - 교통 관련 파생변수 추가
    - `지하철·버스 최단거리`
    - `300m/500m/1km 반경 지하철역/버스정류장 수`
4. 기타 파생 변수
  - 한국기업평판연구소 브랜드평판지수를 기반으로 `아파트명`을 통해 `브랜드등급`(기타, 하이엔드, 프리미엄) 파생변수 추가 
5. 외부 변수 추가
  - 인구수 관련 변수 : `총인구수`, `성비(남/여)` 추가
  - 대출금리 관련 변수 : `loanrate_1m`, `loanrate_3m`, `loanrate_6m`, `loanrate_12m` 추가

## Modeling 전략

- 모델: XGBoost, LightGBM, CatBoost, RandomForest, Stacking, Voting
- 교차검증: 시계열 기반 K-Fold 적용
- 타겟 변수 로그 변환(np.log1p)
- 성능 개선 전략:
    - 성능 낮은 컬럼 제거(Lasso 회귀 기반)
    - 시계열 특성 고려한 학습·검증
    - Voting/Stacking 앙상블 적용
    - AutoInt(DeepCTR) 모델 실험 → 복잡한 비선형 피처 간 상호작용 고려
- **교차 검증**: 시계열 데이터의 특성을 고려하여 **Time Series K-Fold CV**를 적용, 일반화 성능 확보
- **타겟 변환**: 가격(`target`) 분포의 왜도를 줄이기 위해 **로그 변환 (`np.log1p`)** 을 적용
- **앙상블**: 최종적으로 **Voting** 및 **Stacking** 앙상블을 통해 모델의 안정성과 정확도를 높임
- **딥러닝 실험**: **AutoInt** 모델을 도입하여 고차원 피처 상호작용 학습을 시도함

---

## 분석 인사이트 및 결과

- 공변량 시프트 문제 해결:
    - train: 2007~2023.06 / test: 2023.07~09 간 차이
    - train 데이터의 후반부만 사용하여 일반화 성능 확보
- 중요 피처:
    - 전용면적, 계약년월, 연식, 강남3구여부
- 주요 개선:
    - 자치구 대신 좌표 사용
    - 교통변수는 가중치합 사용
    - 모델 성능 향상에 기여하지 못한 컬럼 제거
    - Lasso 회귀로 불필요 피처 제거
- RMSE 개선:
    - 최종 제출 모델 GB_v4 기준 RMSE: **46950.62**

**김장원 – 선형 및 비선형 모델을 통한 인사이트**

- LASSO 회귀를 이용해 변수 중요도를 파악, ‘인구비중’, ‘자치구_구로구’ 등 중요하지 않은 변수 제외
- 시계열 데이터 특성을 고려하지 않은 무작위 K-F출
- K-Means 군집화를 통해 지리적·물리적 유사 아파트 군집 생성
- 전용면적과 건물나이의 상호작용 피처가 가격 예측 핵심임을 확인
- 기존 무작위 데이터 분할에 의한 비현실적 성능 지표 문제를 발견하고 시계열 기반 검증 도입
- 단일 모델 편향성 문제를 인지하고 LightGBM과 CatBoost 앙상블로 예측 안정성 향상
- 모델 설명 가능성 및 신뢰성 확보를 위해 ‘왜’ 예측하는지에 대한 이해 중요성 강조

---

## Evaluation

### RMSE (Root Mean Squared Error)

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

- $N$: 데이터 포인트의 수
- $y_i$: 실제 가격
- $\hat{y}_i$: 모델의 예측 가격

### Meeting Log

- [notion](https://www.notion.so/21d40cb3731d80f18df4e07c93787261?pvs=21)

---

## 📊 결과 및 주요 인사이트

### 최종 결과

- **Evaluation Metric**: RMSE (Root Mean Squared Error)
- **Leader Board Score**: **46950.6270** (5위)

### 핵심 인사이트

1. **공변량 시프트(Covariate Shift) 대응**: 학습 데이터(2007~2023.06)와 테스트 데이터(2023.07~09) 간 분포 차이를 인지하고, **최신 데이터(2013년 10월 이후) 중심의 학습**을 통해 성능을 개선했습니다.
2. **피처의 중요성**: `전용면적`, `계약년월`, `연식`, `강남3구여부`가 예측의 핵심 변수임을 확인했습니다. `자치구` 같은 넓은 범주보다 `좌표`와 `군집` 정보가 더 효과적이었습니다.
3. **모델 설명 가능성**: 모델의 예측 성능만큼 **‘왜’** 그렇게 예측하는지 이해하는 과정이 모델 신뢰도 제고와 개선에 필수적임을 깨달았습니다.
<img width="964" height="1563" alt="image" src="https://github.com/user-attachments/assets/a97de1fd-f6dd-4148-b004-17b3cb60fa64" />
<img width="2163" height="1086" alt="image" src="https://github.com/user-attachments/assets/868116fc-f04a-4cdb-9a5b-3b9b15e669b4" />

<img width="1288" height="754" alt="image" src="https://github.com/user-attachments/assets/1e5a0fa0-ff38-4868-b6c6-7d8bce86ae3e" />
<img width="1362" height="754" alt="image" src="https://github.com/user-attachments/assets/53a6742c-89ec-4508-8e73-8ae14ab0ba7a" />
<img width="2798" height="1447" alt="image" src="https://github.com/user-attachments/assets/8cf2350f-6a19-4fa7-9579-7dc7a59cfee9" />
<img width="1944" height="1172" alt="image" src="https://github.com/user-attachments/assets/3d04c8b2-5bad-43f5-95da-14e5bce708ac" />

---

## 💡 회고 (Retrospective)

### 잘한 점 (What went well)

- **체계적인 협업**: GitHub 중심의 워크플로우와 다양한 협업 도구를 통해 투명하고 효율적인 소통 문화를 구축했습니다.
- **역량 강화**: 팀원 모두가 데이터 처리부터 모델링까지 전 과정을 직접 경험하며 실무 역량을 크게 강화했습니다.

### 아쉬운 점 (What could be improved)

- **시간 관리**: 피처 엔지니어링에 많은 시간을 할애하여 다양한 모델을 깊이 있게 실험할 시간이 부족했습니다.
- **결과 분석**: 최종 제출 전, 예측 결과를 충분히 분석하고 개선할 기회를 놓친 점이 아쉽습니다.

### 팀원별 소감

- **김선민**: 결과 지표에 매몰되기보다, 현상을 깊이 이해하고 ‘왜’를 질문하는 과정의 중요성을 깨달았습니다.
- **김장원**: "Garbage in, Garbage out." 좋은 데이터와 도메인 지식이 모델 성능을 좌우함을 실감했습니다.
- **장윤정**: 전반적인 분석/모델링 경험을 통해, 앞으로 더 다양한 가설을 세우고 실험하는 역량을 키우고 싶습니다.
- **최현화**: 라이브러리 사용을 넘어 모델의 작동 원리를 탐구하고, 문제에 맞는 최적의 해결책을 설계하는 엔지니어가 되겠습니다.

### Reference

- 한국기업평판연구소: https://brikorea.com/
- 주택 관련 논문:
    - https://www.kdi.re.kr/upload/7837/1_1.pdf
    - https://www.emerald.com/insight/content/doi/10.1108/jfm-02-2016-0003/full/html
- *Insert related reference*
