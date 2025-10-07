import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]

wCam, hCam = 840, 840
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

GESTURE_MIN = 50
GESTURE_MAX = 200

def get_hand_landmarks(results, image):
    right, left = None, None
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handType = results.multi_handedness[i].classification[0].label
            lmList = []
            h, w, _ = image.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            if handType == "Right":
                right = lmList
            elif handType == "Left":
                left = lmList
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return right, left

def process_hand_action(image, hand, control_type, range_min, range_max, rect_pos, color, label, callback):
    if hand and len(hand) >= 9:
        x1, y1 = hand[4][1], hand[4][2]
        x2, y2 = hand[8][1], hand[8][2]
        cv2.circle(image, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(image, (x2, y2), 10, (255, 0, 0), -1)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        value = np.interp(length, [GESTURE_MIN, GESTURE_MAX], [range_min, range_max])
        percent = np.interp(length, [GESTURE_MIN, GESTURE_MAX], [0, 100])
        try:
            callback(value)
        except Exception as e:
            print(f"{control_type} control error: {e}")
        x1, y1, x2, y2 = rect_pos
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
        filled_y = int(np.interp(percent, [0, 100], [y2, y1]))
        cv2.rectangle(image, (x1, filled_y), (x2, y2), color, cv2.FILLED)
        cv2.putText(image, f'{label}: {int(percent)}%', (x1 - 10, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            continue
        image = cv2.flip(image, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)
        rightHand, leftHand = get_hand_landmarks(results, image)

        process_hand_action(
            image, rightHand, "Volume",
            minVol, maxVol, (50, 150, 85, 400),
            (0, 255, 0), "Vol", lambda v: volume.SetMasterVolumeLevel(v, None)
        )

        process_hand_action(
            image, leftHand, "Brightness",
            0, 100, (550, 150, 585, 400),
            (255, 255, 0), "Bright", lambda v: sbc.set_brightness(int(v))
        )

        cv2.imshow('Hand Volume & Brightness Control', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
