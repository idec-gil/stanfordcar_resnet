# clothing_classifier

# kagglehub 라이브러리 설치
pip install kagglehub

# kaggle에서 데이터셋 다운로드
python3 importkaggle.py

# brain_mri에 있는 pot, dep txt 파일 불러오기
<img width="475" height="389" alt="image" src="https://github.com/user-attachments/assets/51cd5a34-f51d-4cde-bb91-08c5ffea032b" />
cd ../../brain_mri/value2 로 이동 후 pwd를 누르면 경로 생성
ex)/home/ty/IDEC/brain_mri/value2
이걸 test.py 파일 280 번째 줄에 붙여넣기
ex) LUT_ROOT = "/home/ty/IDEC/brain_mri/value2"

# kaggleimport를 통해 불러온 데이터셋 읽기

ex) "/home/ty/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data/test/"
이걸 271 번째 줄에 복사해서 붙여넣기
