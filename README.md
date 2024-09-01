# Korean_Summarization_B

본 Repository는 말평 2024 일상대화요약 나 유형에서 사용되었습니다.    

해당 코드는 리더보드에서 **3등**을 기록하였으며, 평균점수 61.347, Rouge-1 56.512, Bert score 79.224, BLEURT 48.306 점을 달성하였습니다.   

베이스라인: https://github.com/teddysum/Korean_DCS_2024    
       
**본 모델은 학습 시 A6000 * 6대 이상의 자원이 필요합니다.**    
**추론은 RTX4090 TI * 1대로 가능합니다.**
         
## 사용 방법

### 1. 기본 데이터 삽입

`resource/data` 폴더 내에 다음 데이터를 다운로드하여 삽입합니다.

- **데이터 링크**: [말평 2024 데이터](https://kli.korean.go.kr/benchmark/taskOrdtm/taskDownload.do?taskOrdtmId=147&clCd=END_TASK&subMenuId=sub02)
  - `일상대화요약_train.json`
  - `일상대화요약_dev.json`
  - `일상대화요약_test.json`

### 2. 추가 데이터 삽입 – 일상 대화 요약 말뭉치 2023

1. `resource/data/new_datas` 폴더를 생성합니다.
2. [일상 대화 요약 말뭉치 2023 데이터](https://kli.korean.go.kr/corpus/main/requestMain.do#none)를 다운로드합니다.
   - `NIKL_SDSC_2023_v1.0.zip` 파일을 압축 해제합니다.
3. 압축 해제된 모든 `.json` 파일을 `resource/data/new_datas` 폴더에 삽입합니다.

### 3. 실행 – 필요 라이브러리 설치 및 설정

1. Repository의 경로로 이동합니다.
```bash
cd Korean_Summarization_B/말평2024_일상대화요약_나유형
```

2. 필요 라이브러리를 설치합니다. 
```bash
pip install -r requirements.txt
```

### 4. 실행 – 전처리

1. `newDataPreprocess.ipynb`를 실행하여 추가 데이터 전처리를 수행합니다.

2. `longDataCutter.ipynb`를 실행하여 긴 데이터를 필터링합니다.

3. 다음 코드를 실행합니다. Device 설정이 필요합니다. 
   ```bash
   python run/dataPreprocessor.py

### 5. 실행 – 학습   

다음 명령어를 사용하여 모델 학습을 진행합니다.   
**주의:**  실행 전 device 설정이 필요합니다. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m run.train \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --epoch 2 \
    --lr 2e-5 \
    --warmup_steps 20
```
### 6. 실행 – 추론

학습된 모델을 사용하여 추론을 수행합니다.   
**주의:**  실행 전 device 설정이 필요합니다.
```bash
python -m run.test \
    --output new3.json \
    --model_id maywell/EXAONE-3.0-7.8B-Instruct-Llamafied \
    --device cuda:6 \
    --checkpoint ./resource/results/checkpoint-2874
```
### 7. 후처리
추론 결과를 후처리합니다.    
**주의:**  실행 전 device 설정이 필요합니다.
```bash
python run/resultPostprocessor.py

 ```
### 8. 제출     
후처리된 결과인 new3_1.json 파일을 다운로드하여 제출합니다.
