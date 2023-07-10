# kaggle_digit_recognizer

## 문제 분석

- 0~9 사이의 손으로 그린 숫자 흑백 이미지인 MNIST Dataset의 subset을 예측해서 체출하는 문제이다.

## 데이터 준비

- 각 아미지는 784 pixel이다.
- MNiST이미지는 width, height, channel은 28, 28, 1 이다.
- train.csv에는 783개의 column이 있다.
- test data는 label이 없고, train과 동일한 형태이다.
- test의 pixel 0~783으로 생성한 모델을 실행한 결과를 제출해야 한다.
- 제출파일은 이미지의 id, 라벨로 구성되어야 한다. sample_submission.csv 참고

## 참고 자료

- https://www.kaggle.com/code/vh1981/digit-recognizer-keras
