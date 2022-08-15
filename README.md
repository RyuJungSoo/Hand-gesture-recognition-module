# Hand-gesture-recognition-module
학습시킨 손 모양을 인식해 해당 모션을 따라하는 행동을 하는 아두이노 모듈 제작

## MediaPipe 라이브러리
![hand_tracking_3d_android_gpu](https://user-images.githubusercontent.com/81175672/184478283-bec63c44-f298-4c38-b784-ed9409e510a1.gif)                      
[MediaPipe](https://google.github.io/mediapipe/solutions/hands.html) 라이브러리는 구글에서 만들고, 공개, 관리하고 있는 오픈소스 AI 솔루션이다. 해당 라이브러리는 얼굴 인식, 손 인식, 자세 인식 등 여러 데이터를 감지할 수 있는 기능을 제공한다.        

![hand_landmarks](https://user-images.githubusercontent.com/81175672/184479547-361698dd-362a-44c3-9b23-3e6f08ccf179.png)
MediaPipe에서 **손 랜드마크 모델** 은 직접 좌표 예측을 통해 감지된 손 영역 내부의 21개 3D 손 관절 좌표의 정확한 키포인트 위치를 찾아낸다. 이 모델은 일관된 내부 손 포즈 표현을 학습하고 부분적으로 보이는 손과 자체 폐색에도 견고하다. 
***
그 외에 웹캠으로 입력되는 영상 데이터 처리를 위해 **OpenCV**, 인공지능 모델 학습 및 처리를 위해 **tensorflow** 를 사용한다. 

## 학습 데이터 수집
[학습 데이터 수집 코드](https://github.com/RyuJungSoo/Hand-gesture-recognition-module/blob/main/code/create_dataset.py) 를 사용해 학습 데이터를 수집할 수 있다.   
코드를 실행하면 secs_for_action 동안 각 제스쳐에 대한 데이터를 수집한다. (제스쳐를 바꾸는데 30초의 대기 시간이 있음)               
각 프레임에서 손의 랜드마크가 감지되면 손가락 마디의 랜드마크의 벡터를 사용하여 각도 데이터를 추출한다. 그 후, 랜드마크의 데이터와 라벨링한 각도 데이터를 합친 후 **시퀀스 데이터**로 변형하여 저장한다. (LSTM 사용을 위해)         
## 학습 모델 만들기
[학습 모델 만들기 코드](https://github.com/RyuJungSoo/Hand-gesture-recognition-module/blob/main/code/train.ipynb)                       
연속된 데이터의 처리가 필요하기 때문에 학습 모델로는 **RNN (Recurrent Neural Network)** 를 사용할 것이다.                 
수집한 시퀀스 데이터를 불러와 배열로 만든 다음, 데이터와 라벨을 분리한다. 그 후, 라벨을 One-hot Encoding 시켜준다. (다중 분류 문제이기 때문)          
학습의 정확도를 높이기 위해 train 셋과 test 셋을 분리한 후 학습 모델을 만든다.
사용한 학습 모델은 다음과 같다.
```py
model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]), # input_shape = [30, 99], 30->윈도우의 크기, 99->랜드마크, visibility, 각도
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])
```
학습을 진행하여 **val_acc(검증 정확도)** 가 최적인 모델을 찾아 저장한다.

## 학습 모델 사용
[학습 모델 사용](https://github.com/RyuJungSoo/Hand-gesture-recognition-module/blob/main/code/test.py)              
웹캠으로 들어온 데이터를 **학습 데이터 수집** 과정에 진행한 것처럼 랜드마크의 데이터와 각도 데이터를 합친 데이터를 구한 다음, 학습 모델에 넣어 예측을 진행한다.      
그 후, 연속된 행동이 3번 이상 감지되었을 때 제스쳐를 취했다고 판단하고 해당 제스쳐에 대한 라벨을 화면에 텍스트로 띄워준다.         

## Hand-gesture LED module      
<img src="https://user-images.githubusercontent.com/81175672/184493931-c2a076c8-6691-4513-b15b-be3ef96765ea.jpg"  width="500" height="600"/>            

손 모듈 제작 전에 제스쳐 테스트 용으로, 간단한 LED 회로를 만들어 확인해보았다.                                       
[시연영상](https://youtube.com/shorts/tto-_sStJzU?feature=share)        

## Hand-gesture Hand module
<img src="https://user-images.githubusercontent.com/81175672/184618394-1347f53c-2880-4565-8106-582cc5eab626.jpg"  width="400" height="500"/>         
아두이노 우노와 서보모터를 이용하여 손 모듈의 회로를 제작했다.

## 한계점
- 정교한 모듈이 아니다 보니 즉각적인 모션 반응이 느림.
- 모션에 따른 아두이노의 딜레이를 줄이기 위해 delay를 없애면 노트북과 아두이노가 반응 속도를 따라가지 못해서 멈추는 현상이 일어남. 

# 참고자료
[영상1](https://www.youtube.com/watch?v=CJSobYHYDo4&t=247s)          
[영상2](https://www.youtube.com/watch?v=udeQhZHx-00&t=555s)      
[영상3](https://www.youtube.com/watch?v=eHxDWhtbRCk)
