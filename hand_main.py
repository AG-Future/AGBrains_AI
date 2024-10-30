from asyncore import write
from typing import Tuple
import math
import cv2
import mediapipe as mp
from tensorflow.python.ops.signal.shape_ops import frame

from send_request import send_request


try:
    print("Input stayLength(float number): ", end='')
    stayLength = float(input())
except ValueError:
    stayLength = 1200
    print(f"only Numbers are allowed, now stayLength is default({stayLength})")

# 미디어 파이프의 Hand 모델을 로드합니다.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


HAND_SIZE_THRESHOLD = 0.004
HAND_SIZE_MINI = 0.0823


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def calculate_hand_size():
    # 손목(0번)과 중지 끝(12번) 사이의 거리로 손 크기를 추정
    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[9]
    height, width = frame.shape[:2]

    # 유클리드 거리 계산
    hand_size = length_of((middle.x, middle.y), (wrist.x, wrist.y))
    print(hand_size)

    text = "****WARNING: The distance is too far to be recognized.***"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2  # 프레임의 가로 중앙에 텍스트의 가로 중심 맞추기
    text_y = (height + text_size[1]) // 2  # 프레임의 세로 중앙에 텍스트의 세로 중심 맞추기

    hand_z = abs(abs(hand_landmarks.landmark[9].z) - abs(hand_landmarks.landmark[0].z))
    #print(f'hand_size: {hand_size}\nhand_z: {hand_z}\nhand9z: {hand_landmarks.landmark[9]}\nhand0z: {hand_landmarks.landmark[0].z}')
    if hand_size < HAND_SIZE_MINI:
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    if hand_size < HAND_SIZE_THRESHOLD:
        cv2.putText(frame, "Caution: Too far from camera", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    '''
    hand0z: -8.227418391015817e-08
    hand_size: 0.07266054388026055
    hand_z: 0.0007101459252609743
    hand9z: x: 0.464547366
    y: 0.69831568
    z: 0.00071023294
    
    
    hand0z: 4.253604402038036e-08
    hand_size: 0.01819072473518678
    hand_z: 0.021593765719003954
    hand9z: x: 0.420990974
    y: 0.627887487
    z: -0.0215938296
    '''




def get_direction(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]):
    if length_of(p1, p3) < stayLength or length_of(p2, p3) < stayLength:
        return "stay"
    tilt = get_tilt(p1, p2)

    if tilt == 1j or -1 >= tilt or tilt >= 1:
        if p1[1] > p2[1]:
            return "down"
        else:
            return "up"
    else:
        if p1[0] > p2[0]:
            return "right"
        else:
            return "left"

def get_actual_position(point: Tuple[float, float]) -> Tuple[float, float]:
    return point[0] * cap.get(cv2.CAP_PROP_FRAME_WIDTH), point[1] * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

def length_of(p1: Tuple[float, float], p2: Tuple[float, float]):
    p1 = get_actual_position(p1)
    p2 = get_actual_position(p2)
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2

def get_tilt(p1: Tuple[float, float], p2: Tuple[float, float]):
    p1 = get_actual_position(p1)
    p2 = get_actual_position(p2)
    if p2[0] == p1[0]:
        return 1j
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def x_y():
    x1 = hand_landmarks.landmark[9].x
    y1 = hand_landmarks.landmark[9].y
    #z1 = (hand_landmarks.landmark[12].z * 100) // 1

    x2 = hand_landmarks.landmark[0].x
    y2 = hand_landmarks.landmark[0].y
    #z2 = (hand_landmarks.landmark[0].z * 100) // 1

    direction = get_direction((x1, y1), (x2, y2), (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y))
    '''
    cv2.putText(
        frame,
        f'x1, y1, z1 : {x1}, {y1}, {z1}',
        org=(
            int(hand_landmarks.landmark[12].x * frame.shape[1]), int(hand_landmarks.landmark[12].y * frame.shape[0] + 20)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
        )


    cv2.putText(
        frame,
        f'x2, y2, z2 : {x2}, {y2}, {z2}',
        org=(
            int(hand_landmarks.landmark[0].x * frame.shape[1]),
            int(hand_landmarks.landmark[0].y * frame.shape[0] + 20)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
    )
    '''
    #send_request(direction)

    cv2.putText(
        frame,
        f'direction : {direction}',
        org=(
            int(hand_landmarks.landmark[9].x * frame.shape[1]),
            int(hand_landmarks.landmark[9].y * frame.shape[0] + 20)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=2
    )






with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()  # 변수 이름을 'frame'으로 변경
        if not success:
            continue
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # 프레임을 미디어 파이프에 전달합니다.
        results = hands.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 랜드마크 좌표를 화면에 그립니다.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                calculate_hand_size()
                x_y()



        cv2.imshow('frame', frame)  # 창 이름도 'frame'으로 변경




        if cv2.waitKey(1) == ord('q'):

            break


cap.release()
cv2.destroyAllWindows()

'''
1:-0.010069087147712708
0:1.236920752489823e-07
'''

