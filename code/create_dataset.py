import cv2
import mediapipe as mp
import numpy as np
import time, os

# 학습시킬 데이터 지정
actions = ['zero', 'one', 'two', 'three', 'four', 'five']
seq_length = 30 # window의 사이즈
secs_for_action = 30 # 하나의 제스쳐를 찍는데 걸리는 시간

# MediaPipe hands model

# 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands = 1, # 몇 개의 손을 인식할 것인지
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # 데이터 셋을 저장할 폴더 만들기

# 웹캠을 열어서 데이터 모으기
while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        # 이미지 읽기
        ret, img = cap.read()

        # flip, 웹캠 이미지가 거울처럼 나타나기 때문
        img = cv2.flip(img, 1)

        # 어떤 제스쳐를 학습시킬 것인지 표시
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10,30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
        
        #3초동안 대기
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()
        
        #30초동안 촬영
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv는 기본적으로 BGR로 영상을 읽기 때문에 RGB로 바꿔야 함
            result = hands.process(img) # RGB로 바꾼 데이터를 result에 저장
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21,4))
                    for j,lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # 각 점의 x, y, z 좌표 & 점이 이미지 상에서 보이는지 안 보이는지

                    # 점들 간의 각도 계산하기
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # v2와 v1 사이의 벡터 구하기

                    # 벡터 정규화 시키기(단위 벡터 구하기)
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # 점곱을 구한 다음 arccos으로 각도 구하기
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # 라디안을 각도로 바꾸기

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) # 라벨 추가

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS) # 랜드마크 그리기
            
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'),data)

        # 시퀀스 데이터로 변환
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break






