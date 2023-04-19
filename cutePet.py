import random

import cv2
import mediapipe as mp
import numpy as np
import time

max_num_hands = 1
# gesture = {
#     0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
#     6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
# }
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Training
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

game_start_flag = False
game_start_time = float
last_gesture = 8
com_random_gesture = int
result_text = ""

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles, Normalize
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            # Result
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # GUI
            if idx in rps_gesture.keys():
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Other gestures
            # cv2.putText(img, text=gesture[idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

            # Game
            if game_start_flag == False and idx == 10:
                game_start_flag = True
                game_start_time = time.time()
                com_random_gesture = random.choice([0,5,9])
                cv2.putText(img, text="Game Start!",
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            elif game_start_flag == True:
                if time.time()-game_start_time < 1:
                    cv2.putText(img, text='ROCK~',
                                org=(int(100), int(150)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 0, 0), thickness=4)
                elif time.time() - game_start_time < 2:
                    cv2.putText(img, text='SCISSORS~',
                                org=(int(100), int(150)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 0, 0), thickness=4)
                elif time.time()-game_start_time < 3:
                    img2 = cv2.imread('icon/'+rps_gesture.get(com_random_gesture)+'.png')

                    # cv2.imshow('fw_img', img2)

                    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # cv2.imshow('mask', mask)
                    # cv2.imshow('mask_inv', mask_inv)

                    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

                    height, width, channels = img2.shape
                    roi = img[0:height, 0:width]

                    dst = cv2.addWeighted(roi, 1, img2_fg, 10, 0)
                    img[0:height, 0:width] = dst

                    cv2.putText(img, text='PAPER!',
                                org=(int(100), int(300)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 0, 0), thickness=4)

                    if idx in rps_gesture.keys():
                        last_gesture = idx
                elif time.time()-game_start_time < 6:
                    if last_gesture == 0:
                        if com_random_gesture == 0:
                            result_text = 'Tie'
                        elif com_random_gesture == 5:
                            result_text = 'You Lose!'
                        elif com_random_gesture == 9:
                            result_text = 'You Win!'
                    elif last_gesture == 5:
                        if com_random_gesture == 0:
                            result_text = 'You Win!'
                        elif com_random_gesture == 5:
                            result_text = 'Tie'
                        elif com_random_gesture == 9:
                            result_text = 'You Lose!'
                    elif last_gesture == 9:
                        if com_random_gesture == 0:
                            result_text = 'You Lose!'
                        elif com_random_gesture == 5:
                            result_text = 'You Win!'
                        elif com_random_gesture== 9:
                            result_text = 'Tie'
                    cv2.putText(img, text=result_text, org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 100)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=(0, 0, 255), thickness=3)
                else:
                    game_start_flag = False
                    last_gesture = 8

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Game', img)
    if cv2.waitKey(1) == ord('q'):
        break
