# NogarderLotto - CDM 기법 활용 로또 번호 예측 프로그램

Compound-Dirichlet-Multinomial(CDM) 통계 모델을 활용하여 나눔로또 6/45의 당첨번호를 예측하는 시스템입니다.

## 주요 기능

- CDM 확률 모델을 활용한 로또 번호 예측 알고리즘 구현
- 동행복권 웹사이트에서 과거 당첨 번호 데이터 수집 자동화
- 신뢰성 높은 번호 조합 제안 (기본값: 5개 게임)
- 번호별 확률 시각화 및 결과 저장 기능

## 설치 방법

### 요구사항

- Python 3.8 이상
- 필요 패키지: requests, numpy, scipy, pandas, matplotlib, seaborn

### 설치

1. 저장소 클론
```bash
git clone https://github.com/yourusername/NogarderLotto.git
cd NogarderLotto
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용법

```bash
python src/main.py
```

이 명령어는 기본 설정으로 로또 번호 예측을 실행합니다:
- 첫 회차부터 최신 회차까지의 모든 데이터 수집 및 분석
- CDM 모델 기반 예측 수행
- 5개의 번호 조합 생성 및 출력

### 옵션

```bash
python src/main.py -s 500 -n 10 -v -c -d
```

- `-s`, `--start-draw`: 데이터 수집 시작 회차 (예: 500)
- `-e`, `--end-draw`: 데이터 수집 종료 회차 (기본: 최신 회차)
- `-n`, `--num-combinations`: 생성할 번호 조합 개수 (예: 10)
- `-c`, `--use-cache`: 캐시 사용 여부
- `-o`, `--output-file`: 결과 저장 파일명
- `-f`, `--format`: 결과 저장 형식 (text/json/both)
- `-v`, `--visualize`: 번호 확률 시각화 저장 여부
- `-d`, `--diverse`: 다양성 있는 번호 조합 생성 여부
- `--seed`: 난수 생성 시드

## 프로젝트 구조

```
NogarderLotto/
├── data/                      # 데이터 저장 디렉토리
├── results/                   # 결과 저장 디렉토리
├── src/                       # 소스코드
│   ├── data/                  # 데이터 관련 모듈
│   │   ├── collector.py       # 데이터 수집기
│   │   └── preprocessor.py    # 데이터 전처리기
│   ├── model/                 # 모델 관련 모듈
│   │   ├── cdm_model.py       # CDM 모델 구현
│   │   └── number_generator.py # 번호 생성기
│   ├── utils/                 # 유틸리티 모듈
│   │   ├── helpers.py         # 유틸리티 함수
│   │   └── output_formatter.py # 출력 포맷터
│   ├── main.py                # 메인 프로그램
│   └── test.py                # 테스트 모듈
├── requirements.txt           # 의존성 패키지 목록
└── README.md                  # 프로젝트 설명서
```

## CDM(Compound-Dirichlet-Multinomial) 모델 설명

CDM은 베이지안 통계 기법의 일종으로, 다음의 단계를 통해 로또 번호 예측에 활용됩니다:

1. **사전 분포 설정**: 각 번호(1~45)에 대한 초기 확률 분포 설정
2. **데이터 관측**: 과거 로또 당첨 번호 데이터 수집
3. **사후 분포 계산**: 사전 분포와 관측 데이터를 결합하여 사후 확률 분포 계산
4. **샘플링**: 계산된 사후 분포에서 확률적 샘플링을 통해 번호 조합 생성

이 모델은 과거 데이터의 패턴을 학습하면서도 확률적 특성을 유지하여, 단순한 빈도 분석보다 더 정교한 예측이 가능합니다.

## 주의사항

- 본 프로그램은 통계적 모델을 기반으로 한 예측이며, 당첨을 보장하지 않습니다.
- 로또는 본질적으로 무작위 추첨이므로, 어떤 예측 방법도 100% 정확할 수 없습니다.
- 데이터 수집 시 동행복권 웹사이트의 서버 부하를 줄이기 위해 적절한 딜레이를 두고 요청합니다.

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여하기

버그 보고, 기능 요청, 코드 기여 등은 GitHub 이슈와 풀 리퀘스트를 통해 환영합니다. 