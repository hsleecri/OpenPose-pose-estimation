import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings(action='ignore')


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
# 엑셀 파일 저장을 위한 데이터프레임 생성
pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

columns = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

count_frame = 0
alldata = []
pose_data = []
right_hand_data = []
left_hand_data = []
fps_time = 0
data_temp = {}

# 모델 로드
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(
        min_detection_confidence=0.5)
# 파일 경로
video_path = './미디어파이프 추출/테스트 영상/'
file_name = 'front_2_박스에서 작업물 꺼내기.mp4'
save_path = "./미디어파이프 추출/테스트 영상/"
# 비디오 캡처
cap = cv2.VideoCapture(video_path+file_name)
suc, frame_video = cap.read()

# 프레임별 처리
while True:
    success, image = cap.read()
    if not success:
        break

    # 랜드마크 검출
    image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results_hands = hands.process(image)
    results_pose = pose.process(image)

    # 이미지 분할처리
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_asli = np.copy(image)
    image = np.zeros(image.shape)

    # 랜드마크 그리기
    mp_drawing.draw_landmarks(
            image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    


    # 랜드마크 추출 및 엑셀 데이터 프레임에 추가
    if results_hands.multi_hand_world_landmarks:
        if len(results_hands.multi_hand_world_landmarks) == 2:
            mp_drawing.draw_landmarks(
            image,
            results_hands.multi_hand_world_landmarks[0],
            mp_hands.HAND_CONNECTIONS)
            hand_label = results_hands.multi_handedness[0].classification[0].label
            hand_landmarks = results_hands.multi_hand_world_landmarks[0]
            data_temp = {}
            data_temp.update({"Score":results_hands.multi_handedness[0].classification[0].score})
            
            count=0
            for lm in hand_landmarks.landmark:
                # x, y, z 좌표를 리스트에 추가
                data_temp.update({columns[count]+"_x":lm.x, columns[count]+"_y": lm.y,  columns[count]+"_z": lm.z})
                count+=1
            if hand_label== "Left":
                left_hand_data.append(data_temp)
                hand_landmarks = results_hands.multi_hand_world_landmarks[1]
                data_temp = {}
                data_temp.update({"Score":results_hands.multi_handedness[1].classification[0].score})
                count=0
                for lm in hand_landmarks.landmark:
                    # x, y, z 좌표를 리스트에 추가
                    data_temp.update({columns[count]+"_x":lm.x, columns[count]+"_y": lm.y,  columns[count]+"_z": lm.z})
                    count+=1
                right_hand_data.append(data_temp)
            else :
                right_hand_data.append(data_temp)
                hand_landmarks = results_hands.multi_hand_world_landmarks[1]
                data_temp = {}
                data_temp.update({"Score":results_hands.multi_handedness[1].classification[0].score})
                count=0
                for lm in hand_landmarks.landmark:
                    # x, y, z 좌표를 리스트에 추가
                    data_temp.update({columns[count]+"_x":lm.x, columns[count]+"_y": lm.y,  columns[count]+"_z": lm.z})
                    count+=1
                left_hand_data.append(data_temp)
                    
        elif len(results_hands.multi_hand_world_landmarks) == 1:
            mp_drawing.draw_landmarks(
                image,
                results_hands.multi_hand_world_landmarks[0],
                mp_hands.HAND_CONNECTIONS)

            hand_label = results_hands.multi_handedness[0].classification[0].label
            
            hand_landmarks = results_hands.multi_hand_world_landmarks[0]
            data_temp = {}
            data_temp.update({"Score":results_hands.multi_handedness[0].classification[0].score})
            count=0
            for lm in hand_landmarks.landmark:
                # x, y, z 좌표를 리스트에 추가
                data_temp.update({columns[count]+"_x":lm.x, columns[count]+"_y": lm.y,  columns[count]+"_z": lm.z})
                count += 1
            if hand_label== "Left":
                left_hand_data.append(data_temp)
                # 없는친구
                data_temp = {}
                data_temp.update({"Score":-1})
                for i in range(21):
                    data_temp.update({columns[i]+"_x":-1, columns[i]+"_y": -1,  columns[i]+"_z": -1})  # 추출하지 못한 손 랜드마크는 -1로 채움
                right_hand_data.append(data_temp)
            else :
                right_hand_data.append(data_temp)
                # 없는친구
                data_temp = {}
                data_temp.update({"Score":-1})
                for i in range(21):
                    data_temp.update({columns[i]+"_x":-1, columns[i]+"_y": -1,  columns[i]+"_z": -1})
                left_hand_data.append(data_temp)
    else:
        data_temp = {}
        data_temp.update({"Score":-1})
        for i in range(21):
            data_temp.update({columns[i]+"_x":-1, columns[i]+"_y": -1,  columns[i]+"_z": -1})  # 추출하지 못한 손 랜드마크는 -1로 채움
        right_hand_data.append(data_temp)
        left_hand_data.append(data_temp)
    
    # 포즈 추출
    if results_pose.pose_world_landmarks: 
            data_tubuh = {}
            for i in range(len(pose_tubuh)):
                data_tubuh.update(
                {pose_tubuh[i]+" x" : results_pose.pose_world_landmarks.landmark[i].x,pose_tubuh[i]+" y" : results_pose.pose_world_landmarks.landmark[i].y,pose_tubuh[i]+" z" : results_pose.pose_world_landmarks.landmark[i].z,pose_tubuh[i]+" visiblility" : results_pose.pose_world_landmarks.landmark[i].visibility,}
                )
            pose_data.append(data_tubuh)
    else : 
            data_tubuh = {}
            for i in range(len(pose_tubuh)):
                data_tubuh.update(
                {pose_tubuh[i]+" x" : -1,pose_tubuh[i]+" y" : -1,pose_tubuh[i]+" z" : -1,pose_tubuh[i]+" visiblility" : -1,}
                )
            pose_data.append(data_tubuh)
    
    image = cv2.flip(image, -1)
    image_asli = cv2.flip(image_asli, -1)
    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
    cv2.imshow('MediaPipe Holistic', cv2.resize(image,(0,0),fx=0.1,fy=0.1, interpolation=cv2.INTER_AREA) ) #sudah menampilkan backgrounnd hitam dan skeleton
    cv2.imshow('Gambar asli', cv2.resize(image_asli,(0,0),fx=0.1,fy=0.1, interpolation=cv2.INTER_AREA) )
    count_frame = count_frame + 1
    print(count_frame)
    fps_time = time.time()
    plt.imshow((image*225).astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 엑셀 파일 저장
df_left_hand = pd.DataFrame(left_hand_data)
df_right_hand = pd.DataFrame(right_hand_data)
df_left_hand.columns = ["Left_"+col for col in df_left_hand.columns]
df_right_hand.columns = ["Right_"+col for col in df_right_hand.columns]
hands= {}
hands.update(df_left_hand)
hands.update(df_right_hand)
df_hands=pd.DataFrame(hands)
df_hands.to_excel(
    save_path+file_name+"hands_world_개빡.xlsx")

pose_df = pd.DataFrame(pose_data)
pose_df.to_excel(
    save_path+file_name+"pose_world_개빡.xlsx")
alldata={}
alldata.update(pose_df)
alldata.update(df_left_hand)
alldata.update(df_right_hand)
alldata_df = pd.DataFrame(alldata)
alldata_df.to_excel(
    save_path+file_name+"alldata_world_개빡.xlsx")

cap.release()
cv2.destroyAllWindows()
