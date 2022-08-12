import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['zero', 'one', 'two', 'three', 'four', 'five']
seq_length = 30

model = load_model('models/model.h5')

# MediaPipe hands model (초기화)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

seq = []
action_seq = []

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 점들 간의 각도 계산하기
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # v2와 v1 사이의 벡터 구하기

            # 점곱을 구한 다음 arccos으로 각도 구하기
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # 라디안을 각도로 바꾸기

            d = np.concatenate([joint.flatten(), angle])


            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            # 모델 예측
            y_pred = model.predict(input_data).squeeze()

            # 예측한 값의 인덱스 구하기
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            # confidence가 0.9보다 작으면
            if conf < 0.9:
                continue # 제스쳐 인식 못 한 상황으로 판단

            action = actions[i_pred]
            action_seq.append(action) # action_seq에 action을 저장
            #print(action_seq)

            # 보인 제스쳐의 횟수가 3 미만인 경우에는 계속
            if len(action_seq) < 3:
                continue
            
            # 제스쳐 판단 불가이면 this_action은 ?
            this_action = '?'

            # 만약 마지막 3개의 제스쳐가 같으면 제스쳐가 제대로 취해졌다고 판단
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                print(this_action)

            # 텍스트 출력
            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

