import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# 모델 로드
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 캡처
cap = cv2.VideoCapture(
    "C:/Users/kbk96/Desktop/작업자 데이터 분석/코드/미디어파이프 추출/테스트 영상/skeleton용.mp4")

# 엑셀 파일 저장을 위한 데이터프레임 생성
columns = ['Score','WRIST_x', 'WRIST_y', 'WRIST_z', 'THUMB_CPC_x', 'THUMB_CPC_y', 'THUMB_CPC_z', 'THUMB_MCP_x', 'THUMB_MCP_y', 'THUMB_MCP_z', 'THUMB_IP_x', 'THUMB_IP_y', 'THUMB_IP_z', 'THUMB_TIP_x', 'THUMB_TIP_y', 'THUMB_TIP_z',
           'INDEX_FINGER_MCP_x', 'INDEX_FINGER_MCP_y', 'INDEX_FINGER_MCP_z', 'INDEX_FINGER_PIP_x', 'INDEX_FINGER_PIP_y', 'INDEX_FINGER_PIP_z', 'INDEX_FINGER_DIP_x', 'INDEX_FINGER_DIP_y', 'INDEX_FINGER_DIP_z', 'INDEX_FINGER_TIP_x', 'INDEX_FINGER_TIP_y', 'INDEX_FINGER_TIP_z',
           'MIDDLE_FINGER_MCP_x', 'MIDDLE_FINGER_MCP_y', 'MIDDLE_FINGER_MCP_z', 'MIDDLE_FINGER_PIP_x', 'MIDDLE_FINGER_PIP_y', 'MIDDLE_FINGER_PIP_z', 'MIDDLE_FINGER_DIP_x', 'MIDDLE_FINGER_DIP_y', 'MIDDLE_FINGER_DIP_z', 'MIDDLE_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_y', 'MIDDLE_FINGER_TIP_z',
           'RING_FINGER_PIP_x', 'RING_FINGER_PIP_y', 'RING_FINGER_PIP_z', 'RING_FINGER_DIP_x', 'RING_FINGER_DIP_y', 'RING_FINGER_DIP_z', 'RING_FINGER_TIP_x', 'RING_FINGER_TIP_y', 'RING_FINGER_TIP_z', 'RING_FINGER_MCP_x', 'RING_FINGER_MCP_y', 'RING_FINGER_MCP_z',
           'PINKY_MCP_x', 'PINKY_MCP_y', 'PINKY_MCP_z', 'PINKY_PIP_x', 'PINKY_PIP_y', 'PINKY_PIP_z', 'PINKY_DIP_x', 'PINKY_DIP_y', 'PINKY_DIP_z', 'PINKY_TIP_x', 'PINKY_TIP_y', 'PINKY_TIP_z']
df_left_hand = pd.DataFrame(columns=columns)
df_right_hand = pd.DataFrame(columns=columns)
df_results=pd.DataFrame()

# 프레임별 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 손 검출
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 랜드마크 추출 및 엑셀 데이터 프레임에 추가

    if results.multi_hand_world_landmarks:
        df_results= df_results.append(pd.Series(results), ignore_index=True)
        if len(results.multi_hand_world_landmarks) == 2:
            for hand_idx in range(2):
                hand_label = results.multi_handedness[hand_idx].classification[0].label
                hand_landmarks = results.multi_hand_world_landmarks[hand_idx]
                hand_landmark_row = [results.multi_handedness[hand_idx].classification[0].score]
                for lm in hand_landmarks.landmark:
                    # x, y, z 좌표를 리스트에 추가
                    hand_landmark_row += [lm.x, lm.y, lm.z]
                if hand_label== "Left":
                    df_left_hand = df_left_hand.append(pd.Series(hand_landmark_row,
                                   index=columns), ignore_index=True)
                else :
                    df_right_hand = df_right_hand.append(pd.Series(hand_landmark_row,
                                   index=columns), ignore_index=True)
                    
        elif len(results.multi_hand_world_landmarks) == 1:
            hand_label = results.multi_hand_world_landmarks[0].classification[0].label
            hand_landmarks = results.multi_hand_world_landmarks[0]
            hand_landmark_row = [results.multi_hand_world_landmarks[0].classification[0].score]
            for lm in hand_landmarks.landmark:
                # x, y, z 좌표를 리스트에 추가
                hand_landmark_row += [lm.x, lm.y, lm.z]
            if hand_label== "Left":
                df_left_hand = df_left_hand.append(pd.Series(hand_landmark_row,
                                            index=columns), ignore_index=True)
                # 없는친구
                hand_landmark_row = [-1]
                for _ in range(21):
                    hand_landmark_row += [-1, -1, -1]  # 추출하지 못한 손 랜드마크는 -1로 채움
                df_right_hand = df_right_hand.append(pd.Series(hand_landmark_row,
                                            index=columns), ignore_index=True)
            else :
                df_right_hand = df_right_hand.append(pd.Series(hand_landmark_row,
                                            index=columns), ignore_index=True)
                # 없는친구
                hand_landmark_row = [-1]
                for _ in range(21):
                    hand_landmark_row += [-1, -1, -1]
                df_left_hand = df_left_hand.append(pd.Series(hand_landmark_row,
                                            index=columns), ignore_index=True)
    else:
        hand_landmark_row = [-1]
        for _ in range(21):
            hand_landmark_row += [-1, -1, -1]  # 추출하지 못한 손 랜드마크는 -1로 채움
        df_right_hand = df_right_hand.append(pd.Series(hand_landmark_row,
                       index=columns), ignore_index=True)
        hand_landmark_row = [-1]
        for _ in range(21):
            hand_landmark_row += [-1, -1, -1]  # 추출하지 못한 손 랜드마크는 -1로 채움
        df_left_hand = df_left_hand.append(pd.Series(hand_landmark_row,
                       index=columns), ignore_index=True)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 엑셀 파일 저장
df_left_hand.columns = ["Left_"+col for col in df_left_hand.columns]
df_right_hand.columns = ["Right"+col for col in df_right_hand.columns]
hands= {}

hands.update(df_left_hand)
hands.update(df_right_hand)
df=pd.DataFrame(hands)
df.to_excel(
    "C:/Users/kbk96/Desktop/작업자 데이터 분석/코드/미디어파이프 추출/테스트 영상/hand_landmarks.xlsx", index=False)
df_results.to_excel("C:/Users/kbk96/Desktop/작업자 데이터 분석/코드/미디어파이프 추출/테스트 영상/results.xlsx", index=False)

cap.release()
cv2.destroyAllWindows()
