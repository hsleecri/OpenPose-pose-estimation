import cv2
import mediapipe.python as mp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import time
 
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

count = 0
alldata = []
pose_data = []
right_hand_data = []
left_hand_data = []
fps_time = 0
 
 
pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
 
pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
 
pose_tangan_2 = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
               'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2', 'RING_FINGER_TIP2',
               'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']
video_path = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_RGB\\'
file_name = '4공정_FRONT_CYCLE.mp4'
save_path = "C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_RGB\\"
cap = cv2.VideoCapture(video_path+file_name)
suc,frame_video = cap.read()
#vid_writer = cv2.VideoWriter('C:/Users/SMLC/Desktop/작업자 데이터 분석/결과/skeleton/2.mp4', cv2.VideoWriter_fourcc('H','2','6','4'), 10, (frame_video.shape[1], frame_video.shape[0]))

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            pose_df = pd.DataFrame(pose_data)
            # right_hand_df = pd.DataFrame(right_hand_data)
            # left_hand_df = pd.DataFrame(left_hand_data)
            pose_df.to_excel(save_path+file_name+"pose_world.xlsx")
            # right_hand_df.to_excel(save_path+file_name+"right_hand.xlsx")
            # left_hand_df.to_excel(save_path+file_name+"left_hand.xlsx")
            # df = pd.DataFrame(alldata)
            # df.to_excel(save_path+file_name+"koordinat.xlsx")
            # If loading a video, use 'break' instead of 'continue'.
            break
 
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, -1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)
 
        # Draw landmark annotation on the image.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_asli = np.copy(image)
        image = np.zeros(image.shape)
        # mp_drawing.draw_landmarks(
        #     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # mp_drawing.plot_landmarks(
        # results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
 
    #    if(results.pose_landmarks is not None and results.left_hand_landmarks is not None and results.right_hand_landmarks is not None):
        if results.pose_world_landmarks: 
            data_tubuh = {}
            data_all = {}
            for i in range(len(pose_tubuh)):
                data_tubuh.update(
                {pose_tubuh[i]+" x" : results.pose_world_landmarks.landmark[i].x,pose_tubuh[i]+" y" : results.pose_world_landmarks.landmark[i].y,pose_tubuh[i]+" z" : results.pose_world_landmarks.landmark[i].z,pose_tubuh[i]+" visiblility" : results.pose_world_landmarks.landmark[i].visibility,}
                )
                print(results)
            pose_data.append(data_tubuh)
        #     #오른손
        #     if results.right_hand_landmarks:
        #         data_tangan_kanan = {}
        #         for i in range(len(pose_tangan)):
        #             data_tangan_kanan.update(
        #             {pose_tangan[i]+" x" : results.right_hand_world_landmarks.landmark[i].x,pose_tangan[i]+" y" : results.right_hand_world_landmarks.landmark[i].y,pose_tangan[i]+" z" : results.right_hand_world_landmarks.landmark[i].z,pose_tangan[i]+" visiblility" : results.right_hand_world_landmarks.landmark[i].visibility,}
        #             )
        #         right_hand_data.append(data_tangan_kanan)
        #     else :
        #         data_tangan_kanan = {}
        #         for i in range(len(pose_tangan)):
        #             data_tangan_kanan.update(
        #             {pose_tangan[i]+" x" : -1,pose_tangan[i]+" y" : -1,pose_tangan[i]+" z" : -1,pose_tangan[i]+" visiblility" : -1,}
        #             )
        #         right_hand_data.append(data_tangan_kanan)
        #     #왼손
        #     if results.left_hand_landmarks:
        #         data_tangan_kiri  = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : results.left_hand_world_landmarks.landmark[i].x,pose_tangan_2[i]+" y" : results.left_hand_world_landmarks.landmark[i].y,pose_tangan_2[i]+" z" : results.left_hand_world_landmarks.landmark[i].z,pose_tangan_2[i]+" visiblility" : results.left_hand_world_landmarks.landmark[i].visibility,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
        #     else :
        #         data_tangan_kiri = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : -1,pose_tangan_2[i]+" y" : -1,pose_tangan_2[i]+" z" : -1,pose_tangan_2[i]+" visiblility" :-1,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
        #     data_all.update(data_tubuh)
        #     data_all.update(data_tangan_kanan)
        #     data_all.update(data_tangan_kiri)
        #     alldata.append(data_all)
        # elif results.right_hand_world_landmarks: 
        #     data_all = {}
        #     data_tubuh = {}
        #     for i in range(len(pose_tubuh)):
        #         data_tangan_kanan.update(
        #         {pose_tubuh[i]+" x" : -1,pose_tubuh[i]+" y" : -1,pose_tubuh[i]+" z" : -1,pose_tubuh[i]+" visiblility" : -1,}
        #         )
        #     pose_data.append(data_tubuh)
        #     #오른손
        #     data_tangan_kanan = {}
        #     for i in range(len(pose_tangan)):
        #         data_tangan_kanan.update(
        #         {pose_tangan[i]+" x" : results.right_hand_world_landmarks.landmark[i].x,pose_tangan[i]+" y" : results.right_hand_world_landmarks.landmark[i].y,pose_tangan[i]+" z" : results.right_hand_world_landmarks.landmark[i].z,pose_tangan[i]+" visiblility" : results.right_hand_world_landmarks.landmark[i].visibility,}
        #         )
        #     right_hand_data.append(data_tangan_kanan)
            
        #     #왼손
        #     if results.left_hand_landmarks:
        #         data_tangan_kiri  = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : results.left_hand_world_landmarks.landmark[i].x,pose_tangan_2[i]+" y" : results.left_hand_world_landmarks.landmark[i].y,pose_tangan_2[i]+" z" : results.left_hand_world_landmarks.landmark[i].z,pose_tangan_2[i]+" visiblility" : results.left_hand_world_landmarks.landmark[i].visibility,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
        #     else :
        #         data_tangan_kiri = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : -1,pose_tangan_2[i]+" y" : -1,pose_tangan_2[i]+" z" : -1,pose_tangan_2[i]+" visiblility" :-1,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
        #     data_all.update(data_tubuh)
        #     data_all.update(data_tangan_kanan)
        #     data_all.update(data_tangan_kiri)
        #     alldata.append(data_all)

        else : 
            data_all = {}
            data_tubuh = {}
            for i in range(len(pose_tubuh)):
                data_tubuh.update(
                {pose_tubuh[i]+" x" : -1,pose_tubuh[i]+" y" : -1,pose_tubuh[i]+" z" : -1,pose_tubuh[i]+" visiblility" : -1,}
                )
            pose_data.append(data_tubuh)
        #     #오른손
        #     data_tangan_kanan = {}
        #     for i in range(len(pose_tangan)):
        #         data_tangan_kanan.update(
        #         {pose_tangan[i]+" x" : -1,pose_tangan[i]+" y" : -1,pose_tangan[i]+" z" : -1,pose_tangan[i]+" visiblility" : -1,}
        #         )
        #     right_hand_data.append(data_tangan_kanan)

        #     #왼손
        #     if results.left_hand_landmarks:
        #         data_tangan_kiri  = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : results.left_hand_world_landmarks.landmark[i].x,pose_tangan_2[i]+" y" : results.left_hand_world_landmarks.landmark[i].y,pose_tangan_2[i]+" z" : results.left_hand_world_landmarks.landmark[i].z,pose_tangan_2[i]+" visiblility" : results.left_hand_world_landmarks.landmark[i].visibility,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
        #     else :
        #         data_tangan_kiri = {}
        #         for i in range(len(pose_tangan_2)):
        #             data_tangan_kiri.update(
        #             {pose_tangan_2[i]+" x" : -1,pose_tangan_2[i]+" y" : -1,pose_tangan_2[i]+" z" : -1,pose_tangan_2[i]+" visiblility" :-1,}
        #             )
        #         left_hand_data.append(data_tangan_kiri)
            # data_all.update(data_tubuh)
            # data_all.update(data_tangan_kanan)
            # data_all.update(data_tangan_kiri)
            # alldata.append(data_all)


 
 
        #cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        image = cv2.flip(image, -1)
        image_asli = cv2.flip(image_asli, -1)
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,)
        cv2.imshow('MediaPipe Holistic', cv2.resize(image,(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA) ) #sudah menampilkan backgrounnd hitam dan skeleton
        cv2.imshow('Gambar asli', cv2.resize(image_asli,(0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA) )
        count = count + 1
        print(count)
        fps_time = time.time()
        #vid_writer.write(image)
        # cv2.imshow('MediaPipe Pose',cv2.resize(image,(0,0),fx=0.1,fy=0.1, interpolation=cv2.INTER_AREA) )
        plt.imshow((image*225).astype(np.uint8))
        #plt.savefig("C:/Users/SMLC/Desktop/작업자 데이터 분석/결과/skeleton/" + str(count) + ".jpg")
        if cv2.waitKey(5) & 0xFF == 27:
            #df = pd.DataFrame(alldata)
            #df.to_excel("koordinat.xlsx")
            break
cap.release()