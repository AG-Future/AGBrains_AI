from typing import Tuple
import cv2
import mediapipe as mp
from tensorflow.python.ops.signal.shape_ops import frame


from send_request import send_request



# 미디어 파이프의 Hand 모델을 로드합니다.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



def get_direction(p1: Tuple[int, int], p2: Tuple[int, int]):
    if length_of(p1, p2) < 250:
        return "stay"
    tilt = get_tilt(p1, p2)
    if tilt == 0j or 1.5 <= tilt or tilt <= -1.5:
        if p1[1] > p2[1]:
            return "down"
        else:
            return "up"
    else:
        if p1[0] > p2[0]:
            return "right"
        else:
            return "left"

def length_of(p1: Tuple[int, int], p2: Tuple[int, int]):
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2

def get_tilt(p1: Tuple[int, int], p2: Tuple[int, int]):
    if p2[0] == p1[0]:
        return 0j
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def x_y():
    x1 = (hand_landmarks.landmark[12].x * 100) // 1
    y1 = (hand_landmarks.landmark[12].y * 100) // 1
    z1 = (hand_landmarks.landmark[12].z * 100) // 1

    x2 = (hand_landmarks.landmark[0].x * 100) // 1
    y2 = (hand_landmarks.landmark[0].y * 100) // 1
    z2 = (hand_landmarks.landmark[0].z * 100) // 1

    direction = get_direction((x1, y1), (x2, y2))

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

    send_request(direction)

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
                x_y()


        cv2.imshow('frame', frame)  # 창 이름도 'frame'으로 변경

        if cv2.waitKey(1) == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

