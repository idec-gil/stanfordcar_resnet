# kagglehub 라이브러리 설치
pip install kagglehub

# kaggle에서 데이터셋 다운로드
python3 importkaggle.py

# brain_mri에 있는 pot, dep txt 파일 불러오기
(way1)

<img width="606" height="520" alt="image" src="https://github.com/user-attachments/assets/bebc81ad-d70d-4ad0-9b2a-003cfae787cf" />


(way2)
copy file path를 통해서도 pwd 경로 생성됨

cd ../../brain_mri/value2 로 이동 후 pwd를 누르면 경로 생성
ex)/home/ty/IDEC/brain_mri/value2
이걸 test.py 파일 280 번째 줄에 붙여넣기
ex) LUT_ROOT = "/home/ty/IDEC/brain_mri/value2"

# kaggleimport를 통해 불러온 데이터셋 읽기

위와 방식 동일
ex) "/home/ty/.cache/kagglehub/datasets/jutrera/stanford-car-dataset-by-classes-folder/versions/2/car_data/car_data/test/"
이걸 271 번째 줄에 복사해서 붙여넣기

# 수정 사항
<img width="1230" height="638" alt="image" src="https://github.com/user-attachments/assets/a2f8c969-03cf-4c4e-b10e-39adb98719ad" />

수정사항 적용 후 python3 test2.py
